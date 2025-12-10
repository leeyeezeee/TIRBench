from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os


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