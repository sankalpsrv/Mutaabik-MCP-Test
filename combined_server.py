from typing import Any, Dict, List
import httpx
import os
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from mcp.server import Server
import json
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from wikibaseintegrator.wbi_config import config as wbi_config
from wikibaseintegrator.wbi_helpers import execute_sparql_query
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastMCP server for wikibase functionality
mcp = FastMCP("azure-mcp-server")

# Wikibase Constants
WIKIBASE_BASE_URL = "https://mutaabiklegalresearch.wikibase.cloud"
SPARQL_ENDPOINT = f"{WIKIBASE_BASE_URL}/query/sparql"
WIKIBASE_USER_AGENT = os.getenv("USER_AGENT", "WikibaseMCPTest/1.0")

# Configure wikibaseintegrator
wbi_config['MEDIAWIKI_API_URL'] = 'https://mutaabiklegalresearch.wikibase.cloud/w/api.php'
wbi_config['SPARQL_ENDPOINT_URL'] = "https://mutaabiklegalresearch.wikibase.cloud/query/sparql"
wbi_config['WIKIBASE_URL'] = 'https://mutaabiklegalresearch.wikibase.cloud'
wbi_config['USER_AGENT'] = WIKIBASE_USER_AGENT

# SPARQL examples for few-shot learning
SPARQL_EXAMPLES = [
    {
        "input": "Extract environmental law rule names",
        "query": """PREFIX wd: <https://mutaabiklegalresearch.wikibase.cloud/entity/>
PREFIX wdt: <https://mutaabiklegalresearch.wikibase.cloud/prop/direct/>
PREFIX wdr: <https://mutaabiklegalresearch.wikibase.cloud/prop/reference/>
PREFIX mst: <https://mutaabiklegalresearch.wikibase.cloud/prop/>

SELECT *
WHERE {
wd:Q5 wdt:P5 ?rulenumber .
?rulenumber rdfs:label ?rulesname .
}"""
    },
    {
        "input": "Extract environmental law obligations under the Environment Protection Act, 1986",
        "query": """PREFIX wd: <https://mutaabiklegalresearch.wikibase.cloud/entity/>
PREFIX wdt: <https://mutaabiklegalresearch.wikibase.cloud/prop/direct/>
PREFIX wdr: <https://mutaabiklegalresearch.wikibase.cloud/prop/reference/>
PREFIX mst: <https://mutaabiklegalresearch.wikibase.cloud/prop/>
PREFIX mbq: <https://mutaabiklegalresearch.wikibase.cloud/prop/qualifier/>
PREFIX mbs: <https://mutaabiklegalresearch.wikibase.cloud/prop/statement/>

SELECT ?rulesname ?rulenumber ?obligation_summary 
WHERE {
wd:Q5 wdt:P5 ?rulesnumber .
?rulesnumber rdfs:label ?rulesname .
?rulesnumber mst:P7 ?statementuid .
?statementuid mbs:P7 ?rulenumber .
?statementuid mbq:P11 ?obligation_summary .
}"""
    }
]

# Wikibase helper functions
class WikibaseGraphRAG:
    def __init__(self):
        try:
            # Initialize embeddings and example selector
            self.embeddings = AzureOpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_version="2024-12-01-preview",
                openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT")
            )
            
            # Create example selector using semantic similarity
            self.example_selector = SemanticSimilarityExampleSelector.from_examples(
                SPARQL_EXAMPLES,
                self.embeddings,
                FAISS,
                k=1  # Select only the most similar example
            )
        except Exception as e:
            print(f"Warning: Could not initialize GraphRAG system: {e}")
            self.embeddings = None
            self.example_selector = None

    async def generate_sparql_query(self, natural_language_input: str) -> str:
        """Select the most similar SPARQL query from examples."""
        try:
            if self.example_selector:
                # Select the most similar example
                selected_examples = self.example_selector.select_examples({"input": natural_language_input})
                if selected_examples:
                    return selected_examples[0]["query"]
            # Fallback to first example if no similar match found
            return SPARQL_EXAMPLES[0]["query"]
        except Exception as e:
            raise Exception(f"Failed to select SPARQL query: {str(e)}")

# Initialize the GraphRAG system
graph_rag = WikibaseGraphRAG()

async def execute_wikibase_sparql_query(query: str) -> dict[str, Any] | None:
    """Execute a SPARQL query against the Wikibase instance using wikibaseintegrator."""
    try:
        # Execute the SPARQL query using wikibaseintegrator
        endpoint = SPARQL_ENDPOINT
        user_agent = WIKIBASE_USER_AGENT
        max_retries = 2
        
        # Call the wikibaseintegrator function in a thread since it's synchronous
        import asyncio
        import functools
        
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, 
            functools.partial(
                execute_sparql_query, 
                query=query, 
                endpoint=endpoint, 
                user_agent=user_agent, 
                max_retries=max_retries
            )
        )
        
        return results
    except Exception as e:
        print(f"Error executing SPARQL query: {e}")
        return None

def format_sparql_results(data: dict) -> str:
    """Format SPARQL query results into a readable string."""
    if not data or "results" not in data or "bindings" not in data["results"]:
        return "No results found."
    
    bindings = data["results"]["bindings"]
    if not bindings:
        return "No results found."
    
    # Get variable names from the first binding
    variables = list(bindings[0].keys())
    
    # Format as table
    result_lines = []
    result_lines.append(" | ".join(variables))
    result_lines.append("-" * (len(" | ".join(variables)) + len(variables) * 3))
    
    for binding in bindings[:50]:  # Limit to 50 results for readability
        row = []
        for var in variables:
            if var in binding:
                value = binding[var].get("value", "")
                # Clean up URIs to show just the ID
                if value.startswith("http://www.wikidata.org/entity/"):
                    value = value.split("/")[-1]
                elif value.startswith(WIKIBASE_BASE_URL):
                    value = value.split("/")[-1]
                # Truncate long values
                if len(value) > 60:
                    value = value[:57] + "..."
                row.append(value)
            else:
                row.append("")
        result_lines.append(" | ".join(row))
    
    if len(bindings) > 50:
        result_lines.append(f"\n... and {len(bindings) - 50} more results")
    
    return "\n".join(result_lines)

# Wikibase Tools
@mcp.tool()
async def natural_language_query(question: str) -> str:
    """Convert a natural language question into a SPARQL query, execute it, and return formatted results.
    
    Args:
        question: Natural language question about the Wikibase data
    """
    try:
        # Select SPARQL query from examples
        sparql_query = await graph_rag.generate_sparql_query(question)
        
        # Execute the SPARQL query
        results = await execute_wikibase_sparql_query(sparql_query)
        
        if not results:
            return f"Selected SPARQL query:\n{sparql_query}\n\nError: Unable to execute query or connection failed."
        
        # Format and return results
        formatted_results = format_sparql_results(results)
        
        return f"Selected SPARQL query:\n{sparql_query}\n\nResults:\n{formatted_results}"
        
    except Exception as e:
        return f"Error processing natural language query: {str(e)}"

# Create a standard MCP server for SSE transport
def create_mcp_server():
    server = Server("azure-mcp-server")
    
    @server.list_tools()
    async def handle_list_tools() -> list[dict]:
        return [
            {
                "name": "natural_language_query",
                "description": "Convert a natural language question into a SPARQL query, execute it, and return formatted results",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Natural language question about the Wikibase data"
                        }
                    },
                    "required": ["question"]
                }
            }
        ]
    
    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict) -> list[dict]:
        if name == "natural_language_query":
            question = arguments.get("question", "")
            result = await natural_language_query(question)
            return [{"type": "text", "text": result}]
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    return server

# FastAPI app for HTTP server mode
app = FastAPI(title="Azure MCP Server", description="Wikibase MCP Server")

@app.get("/")
async def root():
    return {"message": "Azure MCP Server running", "tools": ["natural_language_query"]}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/sse")
async def handle_sse():
    """SSE endpoint for MCP transport"""
    async def event_publisher():
        try:
            # Send initial connection event
            yield f"data: {json.dumps({'type': 'connection', 'status': 'connected'})}\n\n"
            
            # Keep connection alive
            while True:
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': asyncio.get_event_loop().time()})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_publisher(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

@app.get("/message")
async def handle_message_get():
    """Handle MCP SSE connection - GET request establishes the stream"""
    async def message_stream():
        try:
            # Send initial connection event
            yield f"data: {json.dumps({'type': 'connection', 'status': 'connected'})}\n\n"
            
            # Keep connection alive with periodic heartbeats
            while True:
                await asyncio.sleep(30)
                yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': asyncio.get_event_loop().time()})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        message_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

@app.post("/message")
async def handle_message_post(request: Request):
    """Handle MCP messages via POST and return JSON response"""
    try:
        message = await request.json()
        
        # Handle different MCP message types
        if message.get("method") == "tools/list":
            tools = [
                {
                    "name": "natural_language_query",
                    "description": "Convert a natural language question into a SPARQL query, execute it, and return formatted results",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Natural language question about the Wikibase data"
                            }
                        },
                        "required": ["question"]
                    }
                }
            ]
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "result": {"tools": tools}
            })
        
        elif message.get("method") == "tools/call":
            params = message.get("params", {})
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if tool_name == "natural_language_query":
                question = arguments.get("question", "")
                result = await natural_language_query(question)
                return JSONResponse(content={
                    "jsonrpc": "2.0",
                    "id": message.get("id"),
                    "result": {
                        "content": [{"type": "text", "text": result}]
                    }
                })
            else:
                return JSONResponse(content={
                    "jsonrpc": "2.0",
                    "id": message.get("id"),
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
                })
        
        else:
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {"code": -32601, "message": f"Unknown method: {message.get('method')}"}
            })
    
    except Exception as e:
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32603, "message": str(e)}
        })


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # Run as HTTP server with SSE support
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Run as MCP stdio server
        mcp.run(transport='stdio')