#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Agent with Tool Calling Support

Supports multiple backends (vLLM, SGLang, Transformers, Remote API) and
optional tool calling capabilities (code execution, web search, mind map).
"""

from __future__ import annotations
import json
import re
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import importlib

# Add agentic_reasoning scripts to path for imports
_AGENTIC_REASONING_PATH = Path(__file__).parent / "agentic_reasoning" / "Agentic-Reasoning" / "scripts"
if _AGENTIC_REASONING_PATH.exists():
    sys.path.insert(0, str(_AGENTIC_REASONING_PATH))


def load_agent_config(config_path: Optional[str] = None) -> 'AgentConfig':
    """
    Load agent configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file. If None, uses default.yaml
    
    Returns:
        AgentConfig instance
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required to load config files. Install with: pip install pyyaml")
    
    if config_path is None:
        # Use default config
        config_path = Path(__file__).parent.parent / "config" / "agent" / "default.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert nested dict to AgentConfig
    # Handle backend-specific configs
    backend = config_dict.get('backend', 'vllm')
    vllm_config = config_dict.get('vllm', {})
    transformers_config = config_dict.get('transformers', {})
    sglang_config = config_dict.get('sglang', {})
    
    # Handle tool configs
    tools_config = config_dict.get('tools', {})
    code_config = tools_config.get('code', {})
    search_config = tools_config.get('search', {})
    mind_map_config = tools_config.get('mind_map', {})
    
    # Handle tool calling config
    tool_calling_config = config_dict.get('tool_calling', {})
    
    # Handle generation config
    generation_config = config_dict.get('generation', {})
    
    return AgentConfig(
        backend=backend,
        model_path=config_dict.get('model_path', ''),
        tokenizer_path=config_dict.get('tokenizer_path'),
        remote_model=config_dict.get('remote_model'),
        
        # Backend-specific
        tensor_parallel_size=vllm_config.get('tensor_parallel_size', 1),
        gpu_memory_utilization=vllm_config.get('gpu_memory_utilization', 0.9),
        device=transformers_config.get('device', 'cuda'),
        device_map=transformers_config.get('device_map'),
        api_base=sglang_config.get('api_base'),
        api_key=sglang_config.get('api_key'),
        
        # Generation
        temperature=generation_config.get('temperature', 0.7),
        top_p=generation_config.get('top_p', 1.0),
        max_tokens=generation_config.get('max_tokens', 2048),
        max_new_tokens=generation_config.get('max_new_tokens', 2048),
        
        # Tools
        enable_tools=tools_config.get('enabled', True),
        tools=tools_config.get('available', []),
        code_model=code_config.get('model'),
        bing_subscription_key=search_config.get('bing_subscription_key'),
        bing_endpoint=search_config.get('bing_endpoint'),
        mind_map_dir=mind_map_config.get('working_dir', './local_mem'),
        
        # Tool calling
        tool_call_format=tool_calling_config.get('format', 'json'),
        max_tool_iterations=tool_calling_config.get('max_iterations', 5),
        system_prompt=config_dict.get('system_prompt'),
    )


# =============== Tool Base Classes ===============
class Tool(ABC):
    """Base class for tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def __call__(self, **kwargs) -> str:
        """Execute the tool with given arguments"""
        pass
    
    def get_tool_description(self) -> Dict[str, Any]:
        """Get tool description in JSON format for prompt"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters()
        }
    
    @abstractmethod
    def _get_parameters(self) -> Dict[str, Any]:
        """Return parameter schema"""
        pass


# =============== Tool Implementations ===============
class CodeAgentTool(Tool):
    """Code generation and execution tool"""
    
    def __init__(self, model_name: Optional[str] = None, working_dir: str = "./tmp"):
        super().__init__(
            name="code_agent",
            description="Generate and execute Python code based on a query. Returns execution results."
        )
        self.model_name = model_name
        self.working_dir = working_dir
        os.makedirs(working_dir, exist_ok=True)
        
        # Initialize code generation model if needed
        if model_name:
            self._init_code_model(model_name)
    
    def _init_code_model(self, model_name: str):
        """Initialize model for code generation"""
        try:
            if model_name in ['gpt-4o', 'claude-3.5-sonnet']:
                from utils.remote_llm import RemoteAPILLM
                self.code_llm = RemoteAPILLM(model_name=model_name)
            else:
                from langchain.chat_models import ChatOpenAI
                self.code_llm = ChatOpenAI(model_name=model_name, temperature=0.7, streaming=True)
        except ImportError as e:
            raise ImportError(f"Failed to import code generation model dependencies: {e}. "
                            f"Make sure agentic_reasoning is properly set up.")
    
    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The code generation query"
                },
                "context": {
                    "type": "string",
                    "description": "Optional context for code generation"
                }
            },
            "required": ["query"]
        }
    
    def __call__(self, query: str, context: str = "") -> str:
        """Generate and execute code"""
        import subprocess
        
        # Generate code
        prompt = f"Given the Context: {context}\n\nWrite a code snippet in Python for the given Problem. Make sure it can be run as a script and directly output the result. OUTPUT JUST CODE SNIPPET AND NOTHING ELSE. Problem: {query}"
        
        if hasattr(self, 'code_llm'):
            if hasattr(self.code_llm, 'invoke'):
                result = self.code_llm.invoke(prompt).content
            else:
                result = self.code_llm.generate([prompt])[0].outputs[0].text
        else:
            # Fallback: return error
            return f"[ERROR] Code generation model not initialized. Please provide model_name."
        
        # Clean markdown
        if "```python" in result:
            result = result[result.find("```python") + 9:result.rfind("```")].strip()
        elif "```" in result:
            result = result[result.find("```") + 3:result.rfind("```")].strip()
        
        # Write and execute
        code_path = os.path.join(self.working_dir, "temp_code.py")
        with open(code_path, "w") as f:
            f.write(result)
        
        try:
            result_obj = subprocess.run(
                ['python', code_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result_obj.stdout if result_obj.returncode == 0 else f"[ERROR] {result_obj.stderr}"
        except subprocess.TimeoutExpired:
            return "[ERROR] Code execution timed out after 10 seconds"


class SearchAgentTool(Tool):
    """Web search tool"""
    
    def __init__(
        self,
        bing_subscription_key: Optional[str] = None,
        bing_endpoint: Optional[str] = None,
        top_k: int = 10,
        use_jina: bool = True,
        jina_api_key: Optional[str] = None
    ):
        super().__init__(
            name="search_agent",
            description="Search the web for information using Bing search engine"
        )
        self.bing_subscription_key = bing_subscription_key
        self.bing_endpoint = bing_endpoint
        self.top_k = top_k
        self.use_jina = use_jina
        self.jina_api_key = jina_api_key
        self.search_cache = {}
        
        # Try to import search agent (not needed for direct function calls)
        self._search_agent_cls = None
    
    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string"
                }
            },
            "required": ["query"]
        }
    
    def __call__(self, query: str) -> str:
        """Execute web search"""
        if query in self.search_cache:
            return f"[CACHED] {self.search_cache[query]}"
        
        if not self.bing_subscription_key or not self.bing_endpoint:
            return "[ERROR] Bing search credentials not provided. Set bing_subscription_key and bing_endpoint."
        
        try:
            from tools.bing_search import bing_web_search, extract_relevant_info
            
            results = bing_web_search(
                query,
                self.bing_subscription_key,
                self.bing_endpoint,
                market='en-US',
                language='en'
            )
            relevant_info = extract_relevant_info(results)[:self.top_k]
            
            # Format results
            formatted = "\n".join([
                f"{i+1}. {info.get('title', '')}\n   {info.get('snippet', '')}\n   URL: {info.get('url', '')}"
                for i, info in enumerate(relevant_info)
            ])
            
            self.search_cache[query] = formatted
            return formatted
        except ImportError as e:
            return f"[ERROR] Failed to import bing_search module: {e}. Make sure agentic_reasoning is set up."
        except Exception as e:
            return f"[ERROR] Search failed: {str(e)}"


class MindMapTool(Tool):
    """Mind map / knowledge graph tool"""
    
    def __init__(self, working_dir: str = "./local_mem", initial_content: str = ""):
        super().__init__(
            name="mind_map",
            description="Query and update a knowledge graph / mind map for structured knowledge retrieval"
        )
        self.working_dir = working_dir
        try:
            from tools.creat_graph import MindMap
            self.mind_map = MindMap(ini_content=initial_content, working_dir=working_dir)
        except ImportError as e:
            self.mind_map = None
            print(f"Warning: MindMap not available: {e}")
    
    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query to search in the knowledge graph"
                },
                "insert": {
                    "type": "string",
                    "description": "Optional content to insert into the graph before querying"
                }
            },
            "required": ["query"]
        }
    
    def __call__(self, query: str, insert: Optional[str] = None) -> str:
        """Query or update mind map"""
        if self.mind_map is None:
            return "[ERROR] MindMap module not available. Please install nano_graphrag."
        
        try:
            # Insert content into graph if provided
            if insert:
                self.mind_map.graph_func.insert(insert)
            
            # Query the graph
            result = self.mind_map.graph_retrieval(query)
            return str(result)
        except Exception as e:
            return f"[ERROR] Mind map query failed: {str(e)}"


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
            return type('RemoteLLM', (), {
                'generate': lambda self, prompts, **kwargs: [
                    type('Output', (), {'outputs': [type('Out', (), {'text': 'response'})()]})()
                    for _ in prompts
                ]
            })()
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


# =============== Tool Call Parser ===============
class ToolCallParser:
    """Parse tool calls from model output"""
    
    @staticmethod
    def parse_json_tool_calls(text: str) -> List[Dict[str, Any]]:
        """Parse JSON-formatted tool calls like: <tool_call>{"name": "code_agent", "arguments": {...}}</tool_call>"""
        tool_calls = []
        
        # Pattern 1: <tool_call>...</tool_call>
        pattern1 = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(pattern1, text, re.DOTALL)
        for match in matches:
            try:
                tool_call = json.loads(match.strip())
                tool_calls.append(tool_call)
            except:
                pass
        
        # Pattern 2: {"tool": "...", "arguments": {...}}
        pattern2 = r'\{"tool"\s*:\s*"[^"]+",\s*"arguments"\s*:\s*\{[^}]+\}\}'
        matches = re.findall(pattern2, text)
        for match in matches:
            try:
                tool_call = json.loads(match)
                tool_calls.append({
                    "name": tool_call.get("tool"),
                    "arguments": tool_call.get("arguments", {})
                })
            except:
                pass
        
        return tool_calls
    
    @staticmethod
    def parse_function_call_format(text: str) -> List[Dict[str, Any]]:
        """Parse function call format like: function_name(arg1="value1", arg2="value2")"""
        tool_calls = []
        
        # Pattern: function_name(...)
        pattern = r'(\w+)\s*\((.*?)\)'
        matches = re.findall(pattern, text)
        
        for func_name, args_str in matches:
            # Simple argument parsing
            args = {}
            # Try to parse as JSON-like or key=value
            for arg_match in re.findall(r'(\w+)\s*=\s*"([^"]+)"', args_str):
                args[arg_match[0]] = arg_match[1]
            
            tool_calls.append({
                "name": func_name,
                "arguments": args
            })
        
        return tool_calls


# =============== Main Agent Class ===============
@dataclass
class AgentConfig:
    """Configuration for LLM Agent"""
    backend: str = "vllm"  # vllm, sglang, transformers, remote_api
    model_path: str = ""
    tokenizer_path: Optional[str] = None
    
    # Backend-specific
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    device: str = "cuda"
    device_map: Optional[str] = None
    
    # API config
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    remote_model: Optional[str] = None  # gpt-4o, claude-3.5-sonnet, etc.
    
    # Generation params
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 2048
    max_new_tokens: int = 2048
    
    # Tool config
    enable_tools: bool = True
    tools: List[str] = field(default_factory=lambda: [])  # ["code", "search", "mind_map"]
    
    # Tool-specific
    code_model: Optional[str] = None
    bing_subscription_key: Optional[str] = None
    bing_endpoint: Optional[str] = None
    mind_map_dir: str = "./local_mem"
    
    # Tool call parsing
    tool_call_format: str = "json"  # json, function_call, auto
    max_tool_iterations: int = 5
    
    # Optional system prompt
    system_prompt: Optional[str] = None


class LLMAgent:
    """
    LLM Agent with optional tool calling capabilities.
    
    Supports multiple backends and can work with or without tools.
    
    Example:
        # From config file
        agent = LLMAgent.from_config("config/agent/vllm_with_tools.yaml")
        
        # Or directly from AgentConfig
        config = AgentConfig(backend="vllm", model_path="/path/to/model")
        agent = LLMAgent(config)
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.backend = self._init_backend()
        self.tokenizer = self.backend.get_tokenizer()
        self.tools: Dict[str, Tool] = {}
        self.conversation_history: List[Dict[str, str]] = []
        self.system_prompt = getattr(config, 'system_prompt', None)
        
        # Initialize tools if enabled
        if config.enable_tools:
            self._init_tools()
    
    @classmethod
    def from_config(cls, config_path: Optional[str] = None) -> 'LLMAgent':
        """Create LLMAgent from YAML config file"""
        config = load_agent_config(config_path)
        return cls(config)
    
    def _init_backend(self) -> LLMBackend:
        """Initialize the appropriate backend"""
        if self.config.backend == "vllm":
            return VLLMBackend(
                model_path=self.config.model_path or self.config.remote_model,
                tokenizer_path=self.config.tokenizer_path,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                sampling_params={
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "max_tokens": self.config.max_tokens
                }
            )
        elif self.config.backend == "sglang":
            api_base = self.config.api_base or os.getenv("SGLANG_API_BASE", "http://127.0.0.1:30000")
            return SGLangBackend(
                api_base=api_base,
                model=self.config.model_path or self.config.remote_model,
                api_key=self.config.api_key or os.getenv("SGLANG_API_KEY")
            )
        elif self.config.backend == "transformers":
            return TransformersBackend(
                model_path=self.config.model_path,
                device=self.config.device,
                device_map=self.config.device_map
            )
        elif self.config.backend == "remote_api":
            if not self.config.remote_model:
                raise ValueError("remote_model must be specified for remote_api backend")
            return RemoteAPIBackend(model_name=self.config.remote_model)
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")
    
    def _init_tools(self):
        """Initialize available tools"""
        tools_to_load = self.config.tools if self.config.tools else ["code", "search", "mind_map"]
        
        for tool_name in tools_to_load:
            if tool_name == "code":
                self.tools["code_agent"] = CodeAgentTool(
                    model_name=self.config.code_model,
                    working_dir="./tmp"
                )
            elif tool_name == "search":
                if self.config.bing_subscription_key and self.config.bing_endpoint:
                    self.tools["search_agent"] = SearchAgentTool(
                        bing_subscription_key=self.config.bing_subscription_key,
                        bing_endpoint=self.config.bing_endpoint,
                        top_k=10
                    )
            elif tool_name == "mind_map":
                self.tools["mind_map"] = MindMapTool(
                    working_dir=self.config.mind_map_dir
                )
    
    def _build_tool_prompt(self) -> str:
        """Build prompt with tool descriptions"""
        if not self.tools:
            return ""
        
        tool_descs = []
        for tool in self.tools.values():
            desc = tool.get_tool_description()
            tool_descs.append(
                f"Tool: {desc['name']}\n"
                f"Description: {desc['description']}\n"
                f"Parameters: {json.dumps(desc['parameters'], indent=2)}\n"
            )
        
        return (
            "\n\nYou have access to the following tools:\n"
            + "\n".join(tool_descs)
            + "\nTo use a tool, format your call as:\n"
            + "<tool_call>{\"name\": \"tool_name\", \"arguments\": {\"arg1\": \"value1\"}}</tool_call>\n"
            + "After tool execution, the results will be provided to you.\n"
        )
    
    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Apply chat template if tokenizer supports it"""
        if self.tokenizer is None:
            # No tokenizer, just concatenate
            return "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                pass
        
        # Fallback
        return "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    
    def _parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Parse tool calls from model output"""
        parser = ToolCallParser()
        
        if self.config.tool_call_format == "json":
            return parser.parse_json_tool_calls(text)
        elif self.config.tool_call_format == "function_call":
            return parser.parse_function_call_format(text)
        else:  # auto
            json_calls = parser.parse_json_tool_calls(text)
            if json_calls:
                return json_calls
            return parser.parse_function_call_format(text)
    
    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool call"""
        if tool_name not in self.tools:
            return f"[ERROR] Tool '{tool_name}' not available. Available tools: {list(self.tools.keys())}"
        
        try:
            tool = self.tools[tool_name]
            result = tool(**arguments)
            return result
        except Exception as e:
            return f"[ERROR] Tool execution failed: {str(e)}"
    
    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        use_tools: Optional[bool] = None,
        max_iterations: Optional[int] = None
    ) -> str:
        """
        Chat with the agent, optionally using tools.
        
        Args:
            message: User message
            system_prompt: Optional system prompt (overrides config default)
            use_tools: Whether to use tools (overrides config)
            max_iterations: Maximum tool call iterations (overrides config)
        
        Returns:
            Final response string
        """
        """
        Chat with the agent, optionally using tools.
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            use_tools: Whether to use tools (overrides config)
            max_iterations: Maximum tool call iterations (overrides config)
        
        Returns:
            Final response string
        """
        use_tools = use_tools if use_tools is not None else self.config.enable_tools
        max_iterations = max_iterations or self.config.max_tool_iterations
        
        # Initialize conversation
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add tool descriptions if using tools
        if use_tools and self.tools:
            tool_prompt = self._build_tool_prompt()
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] += tool_prompt
            else:
                messages.insert(0, {"role": "system", "content": tool_prompt})
        
        # Add conversation history
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": message})
        
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            
            # Generate response
            prompt = self._apply_chat_template(messages)
            response = self.backend.generate(
                [prompt],
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
                max_new_tokens=self.config.max_new_tokens
            )[0]
            
            # Check for tool calls
            if use_tools and self.tools:
                tool_calls = self._parse_tool_calls(response)
                
                if tool_calls:
                    # Add assistant message with tool calls
                    messages.append({"role": "assistant", "content": response})
                    
                    # Execute tools
                    tool_results = []
                    for tool_call in tool_calls:
                        tool_name = tool_call.get("name")
                        arguments = tool_call.get("arguments", {})
                        result = self._execute_tool(tool_name, arguments)
                        tool_results.append(f"Tool: {tool_name}\nResult: {result}\n")
                    
                    # Add tool results as user message
                    tool_results_text = "\n".join(tool_results)
                    messages.append({
                        "role": "user",
                        "content": f"Tool execution results:\n{tool_results_text}\nPlease continue with your response."
                    })
                    
                    # Continue loop to generate final response
                    continue
            
            # No tool calls or tools disabled - return response
            messages.append({"role": "assistant", "content": response})
            self.conversation_history = messages[-10:]  # Keep last 10 messages
            return response
        
        # Max iterations reached
        final_response = messages[-1]["content"] if messages else response
        return f"[WARNING] Maximum tool iterations reached.\n\n{final_response}"
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Simple generation without chat history (bare model mode)"""
        return self.backend.generate(
            [prompt],
            temperature=kwargs.get('temperature', self.config.temperature),
            top_p=kwargs.get('top_p', self.config.top_p),
            max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
            max_new_tokens=kwargs.get('max_new_tokens', self.config.max_new_tokens)
        )[0]
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []