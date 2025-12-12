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

# Import tools and backends from separate modules
from modelloader.tools import Tool, CodeAgentTool, SearchAgentTool, MindMapTool
from modelloader.backends import (
    LLMBackend, VLLMBackend, SGLangBackend, TransformersBackend, RemoteAPIBackend
)

# Add agentic_reasoning scripts to path for imports
_AGENTIC_REASONING_PATH = Path(__file__).parent / "agentic_reasoning" / "Agentic-Reasoning" / "scripts"
if _AGENTIC_REASONING_PATH.exists():
    sys.path.insert(0, str(_AGENTIC_REASONING_PATH))


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
class LLMAgent:
    """
    LLM Agent with optional tool calling capabilities.
    
    Supports multiple backends and can work with or without tools.
    
    Example:
        # Direct initialization from config dictionary
        config = {"backend": "vllm", "model": "/path/to/model", ...}
        agent = LLMAgent(config)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLMAgent from configuration dictionary.
        
        Args:
            config: Configuration dictionary containing all agent settings.
                   Should include backend, model, tools, etc. as loaded by main.py
        """
        self.config = config
        self.backend = self._init_backend()
        self.tokenizer = self.backend.get_tokenizer()
        self.tools: Dict[str, Tool] = {}
        self.conversation_history: List[Dict[str, str]] = []
        
        # Get agent_config for nested access
        agent_config = getattr(config, 'agent_config', None)
        
        self.system_prompt = getattr(agent_config, 'system_prompt', None) or getattr(config, 'system_prompt', None)
        
        # Initialize tools if enabled
        tools_config = getattr(agent_config, 'tools', None)
        enable_tools = getattr(config, 'use_agent_tools', False) and getattr(tools_config, 'enabled', False)
        if enable_tools:
            self._init_tools()
    
    
    def _init_backend(self) -> LLMBackend:
        """Initialize the appropriate backend"""
        backend = getattr(self.config, 'backend', 'vllm')
        agent_config = getattr(self.config, 'agent_config', None)
        
        # Get model path - prefer from agent_config, fallback to top-level
        model_path = getattr(agent_config, 'model_path', None) or getattr(self.config, 'model', '')
        tokenizer_path = getattr(agent_config, 'tokenizer_path', None) or getattr(self.config, 'tokenizer_path', None)
        remote_model = getattr(agent_config, 'remote_model', None) or getattr(self.config, 'remote_model', None)
        
        # Handle sglang model name
        if backend == 'sglang':
            model_path = getattr(self.config, 'sglang_model', None) or model_path
        
        # Get backend-specific configs
        vllm_config = getattr(agent_config, 'vllm', None)
        transformers_config = getattr(agent_config, 'transformers', None)
        sglang_config = getattr(agent_config, 'sglang', None)
        generation_config = getattr(agent_config, 'generation', None)
        
        if backend == "vllm":
            return VLLMBackend(
                model_path=model_path or remote_model,
                tokenizer_path=tokenizer_path,
                tensor_parallel_size=getattr(vllm_config, 'tensor_parallel_size', None) or getattr(self.config, 'tp', 1),
                gpu_memory_utilization=getattr(vllm_config, 'gpu_memory_utilization', None) or getattr(self.config, 'gpu_memory_utilization', 0.9),
                sampling_params={
                    "temperature": getattr(generation_config, 'temperature', None) or getattr(self.config, 'temperature', 0.7),
                    "top_p": getattr(generation_config, 'top_p', None) or getattr(self.config, 'top_p', 1.0),
                    "max_tokens": getattr(generation_config, 'max_tokens', None) or getattr(self.config, 'max_input_tokens', 2048)
                }
            )
        elif backend == "sglang":
            api_base = getattr(sglang_config, 'api_base', None) or getattr(self.config, 'sglang_api_base', None) or os.getenv("SGLANG_API_BASE", "http://127.0.0.1:30000")
            return SGLangBackend(
                api_base=api_base,
                model=model_path or remote_model,
                api_key=getattr(sglang_config, 'api_key', None) or getattr(self.config, 'sglang_api_key', None) or os.getenv("SGLANG_API_KEY")
            )
        elif backend == "transformers":
            return TransformersBackend(
                model_path=model_path,
                device=getattr(transformers_config, 'device', None) or getattr(self.config, 'device', 'cuda'),
                device_map=getattr(transformers_config, 'device_map', None) or getattr(self.config, 'device_map', None)
            )
        elif backend == "remote_api":
            if not remote_model:
                raise ValueError("remote_model must be specified for remote_api backend")
            return RemoteAPIBackend(model_name=remote_model)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _init_tools(self):
        """Initialize available tools"""
        agent_config = getattr(self.config, 'agent_config', None)
        
        tools_config = getattr(agent_config, 'tools', None)
        tools_to_load = getattr(tools_config, 'available', None)
        if not tools_to_load:
            tools_to_load = ["code", "search", "mind_map"]
        
        code_config = getattr(tools_config, 'code', None)
        search_config = getattr(tools_config, 'search', None)
        mind_map_config = getattr(tools_config, 'mind_map', None)
        
        for tool_name in tools_to_load:
            if tool_name == "code":
                self.tools["code_agent"] = CodeAgentTool(
                    model_name=getattr(code_config, 'model', None),
                    working_dir="./tmp"
                )
            elif tool_name == "search":
                bing_key = getattr(search_config, 'bing_subscription_key', None)
                bing_endpoint = getattr(search_config, 'bing_endpoint', None)
                if bing_key and bing_endpoint:
                    self.tools["search_agent"] = SearchAgentTool(
                        bing_subscription_key=bing_key,
                        bing_endpoint=bing_endpoint,
                        top_k=10
                    )
            elif tool_name == "mind_map":
                self.tools["mind_map"] = MindMapTool(
                    working_dir=getattr(mind_map_config, 'working_dir', './local_mem')
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
            + "\nInstructions:\n"
            + "1. You can use tools to help solve the problem when needed.\n"
            + "2. After receiving tool execution results, analyze them carefully.\n"
            + "3. You MUST provide a clear, final answer that directly addresses the original question.\n"
            + "4. For math problems, provide only the final numeric answer.\n"
            + "5. For reading comprehension, provide the answer extracted from the context.\n"
            + "6. Do not include tool calls in your final answer - only provide the answer itself.\n"
        )
    
    def build_prompt_with_dataset(
        self,
        use_tools: Optional[bool] = None,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Build messages that integrate dataset-specific prompt with tool capabilities.
        
        This method:
        1. First adds tool descriptions (if tools are enabled)
        2. Then adds dataset-specific prompt from config
        
        The dataset prompt is loaded directly from config, no template filling is needed.
        
        Args:
            use_tools: Whether to include tool descriptions (defaults to config.use_agent_tools)
            system_prompt: Optional system prompt to override config default
        
        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        agent_config = getattr(self.config, 'agent_config', None)
        tools_config = getattr(agent_config, 'tools', None)
        enable_tools = getattr(self.config, 'use_agent_tools', False) and getattr(tools_config, 'enabled', False)
        use_tools = use_tools if use_tools is not None else enable_tools
        
        # Step 1: Get prompt config from dataset_config
        dataset_config = getattr(self.config, "dataset_config", None)
        prompt_config = getattr(dataset_config, "prompt", None)
        
        # Extract prompt settings from config
        dataset_system = getattr(prompt_config, "system", None) or getattr(prompt_config, "system_prompt", None)
        user_content = getattr(prompt_config, "user_template", "") or getattr(prompt_config, "user", "")
        
        # Step 2: Build system prompt - first add tools, then add dataset system prompt
        system_content = system_prompt or ""
        
        # Add tool descriptions first (if tools are enabled)
        if use_tools and self.tools:
            tool_prompt = self._build_tool_prompt()
            if system_content:
                system_content = system_content + "\n" + tool_prompt
            else:
                system_content = tool_prompt
        
        # Then add dataset-specific system prompt
        if dataset_system:
            if system_content:
                system_content = system_content + "\n" + dataset_system
            else:
                system_content = dataset_system
        
        # Step 3: Build messages structure
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        if user_content:
            messages.append({"role": "user", "content": user_content})
        
        return messages
    
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
        agent_config = getattr(self.config, 'agent_config', None)
        tool_calling_config = getattr(agent_config, 'tool_calling', None)
        tool_call_format = getattr(tool_calling_config, 'format', 'json')
        
        if tool_call_format == "json":
            return parser.parse_json_tool_calls(text)
        elif tool_call_format == "function_call":
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
        message: str = "",
        system_prompt: Optional[str] = None,
        use_tools: Optional[bool] = None,
        max_iterations: Optional[int] = None,
        example: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Chat with the agent, optionally using tools and dataset-specific prompts.
        
        Args:
            message: User message (ignored if example is provided)
            system_prompt: Optional system prompt (overrides config default)
            use_tools: Whether to use tools (overrides config)
            max_iterations: Maximum tool call iterations (overrides config)
            example: Optional example dict (used with dataset config to build prompt)
        
        Returns:
            Final response string
        """
        agent_config = getattr(self.config, 'agent_config', None)
        tools_config = getattr(agent_config, 'tools', None)
        tool_calling_config = getattr(agent_config, 'tool_calling', None)
        generation_config = getattr(agent_config, 'generation', None)
        
        enable_tools = getattr(self.config, 'use_agent_tools', False) and getattr(tools_config, 'enabled', False)
        use_tools = use_tools if use_tools is not None else enable_tools
        max_iterations = max_iterations or getattr(tool_calling_config, 'max_iterations', 5)
        
        # Build initial messages
        dataset_config = getattr(self.config, "dataset_config", None)
        if dataset_config:
            # Use dataset-specific prompt building from config
            messages = self.build_prompt_with_dataset(
                use_tools=use_tools,
                system_prompt=system_prompt
            )
        else:
            # Traditional message-based approach
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add tool descriptions if using tools
            if use_tools and self.tools:
                tool_prompt = self._build_tool_prompt()
                if messages and messages[0]["role"] == "system":
                    messages[0]["content"] += "\n" + tool_prompt
                else:
                    messages.insert(0, {"role": "system", "content": tool_prompt})
            
            # Add conversation history
            messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": message})
        
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            
            # Generate response - unified logic
            prompt = self._apply_chat_template(messages)
            response = self.backend.generate(
                [prompt],
                temperature=getattr(generation_config, 'temperature', None) or getattr(self.config, 'temperature', 0.7),
                top_p=getattr(generation_config, 'top_p', None) or getattr(self.config, 'top_p', 1.0),
                max_tokens=getattr(generation_config, 'max_tokens', None) or getattr(self.config, 'max_input_tokens', 2048),
                max_new_tokens=getattr(generation_config, 'max_new_tokens', None) or getattr(self.config, 'max_new_tokens', 2048)
            )[0]
            
            # Check for tool calls - unified logic
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
                    
                    # Add tool results as user message with instruction to provide final answer
                    tool_results_text = "\n".join(tool_results)
                    final_answer_instruction = (
                        "Based on the tool execution results above, please provide your final answer. "
                        "The answer should be clear, concise, and directly address the original question. "
                        "If this is a math problem, provide only the final numeric answer. "
                        "If this is a reading comprehension question, provide the answer from the context."
                    )
                    messages.append({
                        "role": "user",
                        "content": f"Tool execution results:\n{tool_results_text}\n{final_answer_instruction}"
                    })
                    
                    # Continue loop to generate final response
                    continue
            
            # No tool calls or tools disabled - return response
            messages.append({"role": "assistant", "content": response})
            self.conversation_history = messages[-10:]  # Keep last 10 messages
            return response
        
        # Max iterations reached - extract final answer from last response
        final_response = messages[-1]["content"] if messages[-1].get("role") == "assistant" else response
        
        # Try to extract a clean answer if response contains tool calls or warnings
        if "<tool_call>" in final_response or "[WARNING]" in final_response:
            # Extract the last meaningful part after tool calls
            parts = final_response.split("</tool_call>")
            if len(parts) > 1:
                final_response = parts[-1].strip()
        
        return final_response
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Simple generation without chat history (bare model mode)"""
        agent_config = getattr(self.config, 'agent_config', None)
        generation_config = getattr(agent_config, 'generation', None)
        
        return self.backend.generate(
            [prompt],
            temperature=kwargs.get('temperature', getattr(generation_config, 'temperature', None) or getattr(self.config, 'temperature', 0.7)),
            top_p=kwargs.get('top_p', getattr(generation_config, 'top_p', None) or getattr(self.config, 'top_p', 1.0)),
            max_tokens=kwargs.get('max_tokens', getattr(generation_config, 'max_tokens', None) or getattr(self.config, 'max_input_tokens', 2048)),
            max_new_tokens=kwargs.get('max_new_tokens', getattr(generation_config, 'max_new_tokens', None) or getattr(self.config, 'max_new_tokens', 2048))
        )[0]
    
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []