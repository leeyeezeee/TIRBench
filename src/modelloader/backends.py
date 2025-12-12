from abc import ABC, abstractmethod
from typing import List, Optional


# =============== LLM Backend Interfaces ===============
class LLMBackend(ABC):
    """Abstract base class for LLM backends"""
    
    @abstractmethod
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text from prompts"""
        pass
    
    @abstractmethod
    def get_tokenizer(self):
        """Get tokenizer for chat template"""
        pass


class VLLMBackend(LLMBackend):
    """vLLM backend"""
    
    def __init__(self, model_path: str, tokenizer_path: Optional[str] = None, **kwargs):
        from vllm import LLM, SamplingParams
        
        self.llm = LLM(
            model=model_path,
            tokenizer=tokenizer_path or model_path,
            **kwargs
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.default_params = SamplingParams(**kwargs.get('sampling_params', {}))
    
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        from vllm import SamplingParams
        
        params = kwargs.get('sampling_params', self.default_params)
        if isinstance(params, dict):
            params = SamplingParams(**params)
        
        outputs = self.llm.generate(prompts, params)
        return [out.outputs[0].text for out in outputs]
    
    def get_tokenizer(self):
        return self.tokenizer


class SGLangBackend(LLMBackend):
    """SGLang backend (via API)"""
    
    def __init__(self, api_base: str, model: str, api_key: Optional[str] = None):
        import requests
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        
        # Try to load tokenizer locally
        self.tokenizer = None
        try:
            from transformers import AutoTokenizer
            # Use a reasonable default tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except:
            pass
    
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        import requests
        import json
        import concurrent.futures
        
        temperature = kwargs.get('temperature', 0.7)
        top_p = kwargs.get('top_p', 1.0)
        max_tokens = kwargs.get('max_tokens', 2048)
        concurrency = kwargs.get('concurrency', 8)
        
        def _generate_one(prompt: str) -> str:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens
            }
            r = requests.post(
                f"{self.api_base}/v1/chat/completions",
                headers=self.headers,
                data=json.dumps(payload),
                timeout=120
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            results = list(executor.map(_generate_one, prompts))
        return results
    
    def get_tokenizer(self):
        return self.tokenizer


class TransformersBackend(LLMBackend):
    """Transformers backend"""
    
    def __init__(self, model_path: str, device: str = "cuda", device_map: Optional[str] = None, **kwargs):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=kwargs.get('local_files_only', False)
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model_kwargs = dict(trust_remote_code=True, local_files_only=kwargs.get('local_files_only', False))
        if device_map:
            model_kwargs["device_map"] = device_map
        
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        
        if not device_map:
            if device.startswith("cuda"):
                self.model = self.model.to("cuda")
            elif device.startswith("npu"):
                import torch_npu
                self.model = self.model.to(device)
            else:
                self.model = self.model.to("cpu")
        
        self.device = device
        self.device_map = device_map
    
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        import torch
        
        max_new_tokens = kwargs.get('max_new_tokens', 2048)
        temperature = kwargs.get('temperature', 0.7)
        do_sample = kwargs.get('do_sample', False)
        
        results = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            if not self.device_map:
                inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            results.append(text)
        return results
    
    def get_tokenizer(self):
        return self.tokenizer


class RemoteAPIBackend(LLMBackend):
    """Remote API backend (OpenAI/Anthropic)"""
    
    def __init__(self, model_name: str):
        try:
            from utils.remote_llm import RemoteAPILLM
            self.llm = RemoteAPILLM(model_name=model_name)
        except ImportError:
            # Fallback implementation
            self.llm = self._create_remote_llm(model_name)
        
        # Tokenizer for chat template
        from transformers import AutoTokenizer
        if 'gpt' in model_name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        elif 'claude' in model_name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Placeholder
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    def _create_remote_llm(self, model_name: str):
        """Create remote LLM if import fails"""
        import os
        if 'gpt' in model_name.lower():
            from openai import OpenAI
            
            # Get API key and base URL from environment
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            
            # Create OpenAI client
            client_kwargs = {}
            if api_key:
                client_kwargs["api_key"] = api_key
            if base_url:
                client_kwargs["base_url"] = base_url
            
            client = OpenAI(**client_kwargs) if client_kwargs else OpenAI()
            
            class RemoteLLM:
                def __init__(self, client, model_name):
                    self.client = client
                    self.model_name = model_name
                
                def generate(self, prompts, sampling_params=None):
                    from dataclasses import dataclass
                    if sampling_params is None:
                        @dataclass
                        class SamplingParams:
                            max_tokens: int = 2000
                            temperature: float = 0.7
                            top_p: float = 0.8
                        sampling_params = SamplingParams()
                    
                    results = []
                    for prompt in prompts:
                        try:
                            # Convert prompt to messages format if needed
                            if isinstance(prompt, str):
                                # Try to parse as chat template or use as single message
                                messages = [{"role": "user", "content": prompt}]
                            else:
                                messages = prompt
                            
                            response = self.client.chat.completions.create(
                                model=self.model_name,
                                messages=messages,
                                max_tokens=getattr(sampling_params, 'max_tokens', 2000),
                                temperature=getattr(sampling_params, 'temperature', 0.7),
                                top_p=getattr(sampling_params, 'top_p', 0.8)
                            )
                            
                            text = response.choices[0].message.content
                            results.append(type('Output', (), {
                                'outputs': [type('Out', (), {'text': text})()]
                            })())
                        except Exception as e:
                            # Fallback on error
                            results.append(type('Output', (), {
                                'outputs': [type('Out', (), {'text': f'Error: {str(e)}'})()]
                            })())
                    
                    return results
            
            return RemoteLLM(client, model_name)
        return None
    
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        try:
            from utils.remote_llm import SamplingParams
        except ImportError:
            # Fallback SamplingParams
            from dataclasses import dataclass
            @dataclass
            class SamplingParams:
                max_tokens: int = 2000
                temperature: float = 0.7
                top_p: float = 0.8
                stop: Optional[List[str]] = None
        
        params = SamplingParams(
            max_tokens=kwargs.get('max_tokens', 2048),
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 1.0),
            stop=kwargs.get('stop', None)
        )
        
        outputs = self.llm.generate(prompts, sampling_params=params)
        return [out.outputs[0].text for out in outputs]
    
    def get_tokenizer(self):
        return self.tokenizer