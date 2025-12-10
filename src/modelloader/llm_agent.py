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
        # Direct initialization from AgentConfig
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
        example: Dict[str, Any],
        dataset_loader: Any,
        use_tools: Optional[bool] = None,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Build messages that integrate dataset-specific formatting with tool capabilities.
        
        This method combines:
        1. Dataset-specific prompt from dataset_loader.build_prompt
        2. Tool descriptions (if tools are enabled)
        
        Args:
            example: Example dict with 'question', 'context', etc.
            dataset_loader: DatasetLoader instance for dataset-specific prompt building
            use_tools: Whether to include tool descriptions (defaults to config.enable_tools)
            system_prompt: Optional system prompt to override config default
        
        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        use_tools = use_tools if use_tools is not None else self.config.enable_tools
        
        # Step 1: Get base prompt from dataset_loader.build_prompt
        base_prompt = dataset_loader.build_prompt(example, self.tokenizer)
        
        # Step 2: Build system prompt with tool descriptions if needed
        system_content = system_prompt or self.system_prompt or ""
        
        if use_tools and self.tools:
            tool_prompt = self._build_tool_prompt()
            if system_content:
                system_content = system_content + "\n" + tool_prompt
            else:
                # Default system prompt when tools are enabled
                system_content = (
                    "You are a helpful assistant that can use tools to solve problems. "
                    "Use tools when needed, but always provide a clear final answer.\n"
                    + tool_prompt
                )
        
        # Step 3: Build messages structure
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": base_prompt})
        
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
        message: str = "",
        system_prompt: Optional[str] = None,
        use_tools: Optional[bool] = None,
        max_iterations: Optional[int] = None,
        dataset_loader: Any = None,
        example: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Chat with the agent, optionally using tools and dataset-specific prompts.
        
        Args:
            message: User message (ignored if dataset_loader and example are provided)
            system_prompt: Optional system prompt (overrides config default)
            use_tools: Whether to use tools (overrides config)
            max_iterations: Maximum tool call iterations (overrides config)
            dataset_loader: Optional DatasetLoader for dataset-specific prompt building
            example: Optional example dict (used with dataset_loader to build prompt)
        
        Returns:
            Final response string
        """
        use_tools = use_tools if use_tools is not None else self.config.enable_tools
        max_iterations = max_iterations or self.config.max_tool_iterations
        
        # Build initial messages - unified approach
        if dataset_loader and example:
            # Use dataset-specific prompt building
            messages = self.build_prompt_with_dataset(
                example=example,
                dataset_loader=dataset_loader,
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
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
                max_new_tokens=self.config.max_new_tokens
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
        return self.backend.generate(
            [prompt],
            temperature=kwargs.get('temperature', self.config.temperature),
            top_p=kwargs.get('top_p', self.config.top_p),
            max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
            max_new_tokens=kwargs.get('max_new_tokens', self.config.max_new_tokens)
        )[0]
    
    def solve_with_dataset(
        self,
        example: Dict[str, Any],
        dataset_loader: Any,
        use_tools: Optional[bool] = None,
        max_iterations: Optional[int] = None
    ) -> str:
        """
        Solve a dataset example using dataset-specific prompt formatting and optional tools.
        
        This is a convenience method that combines build_prompt_with_dataset and chat
        to provide a simple interface for solving dataset examples.
        
        Args:
            example: Example dict with 'question', 'context', etc.
            dataset_loader: DatasetLoader instance for dataset-specific prompt building
            use_tools: Whether to use tools (defaults to config.enable_tools)
            max_iterations: Maximum tool call iterations (defaults to config.max_tool_iterations)
        
        Returns:
            Final answer string
        """
        return self.chat(
            message="",  # Will be built from example and dataset_loader
            use_tools=use_tools,
            max_iterations=max_iterations,
            dataset_loader=dataset_loader,
            example=example
        )
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []