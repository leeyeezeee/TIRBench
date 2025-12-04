# LLM Agent Configuration

This directory contains configuration files for the LLM Agent with tool calling capabilities.

## Configuration Files

- `default.yaml`: Default configuration template
- `vllm_with_tools.yaml`: vLLM backend with all tools enabled
- `remote_api_bare.yaml`: Remote API (OpenAI/Anthropic) without tools
- `sglang_with_code.yaml`: SGLang backend with code execution tool

## Usage

### From Config File

```python
from modelloader.llm_agent import LLMAgent

# Load from config file
agent = LLMAgent.from_config("config/agent/vllm_with_tools.yaml")

# Use the agent
response = agent.chat("Search for Python examples and write code to test them")
print(response)
```

### Direct Configuration

```python
from modelloader.llm_agent import LLMAgent, AgentConfig

# Create config directly
config = AgentConfig(
    backend="vllm",
    model_path="/path/to/model",
    enable_tools=True,
    tools=["code", "search"],
    bing_subscription_key="your_key",
    bing_endpoint="https://api.bing.microsoft.com/v7.0/search"
)

agent = LLMAgent(config)
response = agent.chat("Your question here")
```

## Configuration Options

### Backend Selection

- `vllm`: Local vLLM backend (requires model path)
- `sglang`: SGLang API backend
- `transformers`: HuggingFace Transformers backend
- `remote_api`: OpenAI/Anthropic API backend

### Tools

Available tools:
- `code`: Code generation and execution
- `search`: Web search via Bing API
- `mind_map`: Knowledge graph queries

### Environment Variables

Some settings can be overridden via environment variables:
- `SGLANG_API_BASE`: SGLang API base URL
- `SGLANG_API_KEY`: SGLang API key
- `OPENAI_API_KEY`: OpenAI API key (for remote_api or code tool)
- `ANTHROPIC_API_KEY`: Anthropic API key (for remote_api or code tool)

## Example Configurations

### vLLM with Tools

```yaml
backend: "vllm"
model_path: "/path/to/your/model"
tools:
  enabled: true
  available: ["code", "search", "mind_map"]
```

### Remote API (No Tools)

```yaml
backend: "remote_api"
remote_model: "gpt-4o"
tools:
  enabled: false
```

### SGLang with Code Tool

```yaml
backend: "sglang"
model_path: "Qwen/Qwen2.5-7B-Instruct"
sglang:
  api_base: "http://127.0.0.1:30000"
tools:
  enabled: true
  available: ["code"]
```

