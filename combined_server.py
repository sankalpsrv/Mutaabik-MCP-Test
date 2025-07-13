"""
Mutaabik MCP Server - Simplified Wikibase SPARQL Query Server
Provides natural language to SPARQL query functionality for legal/environmental data.
"""

import os
import asyncio
import functools
from typing import Any, Dict, Optional

# Core dependencies
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastmcp import FastMCP
import json

# AI and embeddings
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

# Wikibase integration
from wikibaseintegrator.wbi_config import config as wbi_config
from wikibaseintegrator.wbi_helpers import execute_sparql_query

# Environment
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
WIKIBASE_BASE_URL = "https://mutaabiklegalresearch.wikibase.cloud"
SPARQL_ENDPOINT = f"{WIKIBASE_BASE_URL}/query/sparql"
USER_AGENT = os.getenv("USER_AGENT", "MutaabikMCP/1.0")

# Configure wikibaseintegrator
wbi_config['MEDIAWIKI_API_URL'] = f'{WIKIBASE_BASE_URL}/w/api.php'
wbi_config['SPARQL_ENDPOINT_URL'] = SPARQL_ENDPOINT
wbi_config['WIKIBASE_URL'] = WIKIBASE_BASE_URL
wbi_config['USER_AGENT'] = USER_AGENT

# SPARQL query examples for semantic matching
SPARQL_EXAMPLES = [
    {
        "input": "Extract environmental law rule names",
        "query": """PREFIX wd: <https://mutaabiklegalresearch.wikibase.cloud/entity/>
PREFIX wdt: <https://mutaabiklegalresearch.wikibase.cloud/prop/direct/>
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

# Initialize AI components
def create_embeddings() -> Optional[AzureOpenAIEmbeddings]:
    """Create Azure OpenAI embeddings if credentials are available."""
    try:
        return AzureOpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_version="2024-12-01-preview",
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT")
        )
    except Exception as e:
        print(f"Warning: Could not initialize Azure OpenAI embeddings: {e}")
        return None

def create_example_selector(embeddings: Optional[AzureOpenAIEmbeddings]) -> Optional[SemanticSimilarityExampleSelector]:
    """Create semantic similarity example selector if embeddings are available."""
    if not embeddings:
        return None
    
    try:
        return SemanticSimilarityExampleSelector.from_examples(
            SPARQL_EXAMPLES,
            embeddings,
            FAISS,
            k=1
        )
    except Exception as e:
        print(f"Warning: Could not initialize example selector: {e}")
        return None

# Initialize components
embeddings = create_embeddings()
example_selector = create_example_selector(embeddings)

def select_sparql_query(question: str) -> str:
    """Select the most appropriate SPARQL query for the given question."""
    if example_selector:
        try:
            examples = example_selector.select_examples({"input": question})
            if examples:
                return examples[0]["query"]
        except Exception as e:
            print(f"Error selecting SPARQL query: {e}")
    
    # Fallback to first example
    return SPARQL_EXAMPLES[0]["query"]

async def execute_sparql_async(query: str) -> Optional[Dict[str, Any]]:
    """Execute SPARQL query asynchronously using wikibaseintegrator."""
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(
                execute_sparql_query,
                query=query,
                endpoint=SPARQL_ENDPOINT,
                user_agent=USER_AGENT,
                max_retries=2
            )
        )
    except Exception as e:
        print(f"Error executing SPARQL query: {e}")
        return None

def format_results(data: Optional[Dict[str, Any]]) -> str:
    """Format SPARQL results into a readable table."""
    if not data or "results" not in data or "bindings" not in data["results"]:
        return "No results found."
    
    bindings = data["results"]["bindings"]
    if not bindings:
        return "No results found."
    
    # Get variable names
    variables = list(bindings[0].keys())
    
    # Create table
    lines = [
        " | ".join(variables),
        "-" * (len(" | ".join(variables)) + len(variables) * 3)
    ]
    
    # Add data rows (limit to 50 for readability)
    for binding in bindings[:50]:
        row = []
        for var in variables:
            if var in binding:
                value = binding[var].get("value", "")
                # Clean up URIs
                if value.startswith(("http://www.wikidata.org/entity/", WIKIBASE_BASE_URL)):
                    value = value.split("/")[-1]
                # Truncate long values
                if len(value) > 60:
                    value = value[:57] + "..."
                row.append(value)
            else:
                row.append("")
        lines.append(" | ".join(row))
    
    if len(bindings) > 50:
        lines.append(f"\n... and {len(bindings) - 50} more results")
    
    return "\\n".join(lines)

# Initialize FastMCP server
mcp = FastMCP("mutaabik-mcp-server")

@mcp.tool()
async def natural_language_query(question: str) -> str:
    """Convert natural language question to SPARQL query and execute it.
    
    Args:
        question: Natural language question about the Wikibase data
        
    Returns:
        Formatted results including the SPARQL query used and data table
    """
    try:
        # Select appropriate SPARQL query
        sparql_query = select_sparql_query(question)
        
        # Execute query
        results = await execute_sparql_async(sparql_query)
        
        if not results:
            return f"Query: {sparql_query}\\n\\nError: Unable to execute query or connection failed."
        
        # Format results
        formatted_results = format_results(results)
        
        return f"Query: {sparql_query}\\n\\nResults:\\n{formatted_results}"
        
    except Exception as e:
        return f"Error processing query: {str(e)}"

# FastAPI app for HTTP mode
app = FastAPI(title="Mutaabik MCP Server", description="Wikibase SPARQL Query Server")

@app.get("/")
async def root():
    return {"message": "Mutaabik MCP Server", "tools": ["natural_language_query"]}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/message")
async def message_sse():
    """SSE endpoint for MCP inspector connection."""
    async def stream():
        yield f"data: {json.dumps({'type': 'connection', 'status': 'connected'})}\\n\\n"
        while True:
            await asyncio.sleep(30)
            yield f"data: {json.dumps({'type': 'heartbeat'})}\\n\\n"
    
    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )

@app.post("/message")
async def message_post(request: Request):
    """Handle MCP JSON-RPC messages."""
    try:
        message = await request.json()
        method = message.get("method")
        
        if method == "tools/list":
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "result": {
                    "tools": [{
                        "name": "natural_language_query",
                        "description": "Convert natural language question to SPARQL query and execute it",
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
                    }]
                }
            })
        
        elif method == "tools/call":
            params = message.get("params", {})
            if params.get("name") == "natural_language_query":
                question = params.get("arguments", {}).get("question", "")
                result = await natural_language_query(question)
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": message.get("id"),
                    "result": {"content": [{"type": "text", "text": result}]}
                })
            else:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": message.get("id"),
                    "error": {"code": -32601, "message": f"Unknown tool: {params.get('name')}"}
                })
        
        else:
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {"code": -32601, "message": f"Unknown method: {method}"}
            })
    
    except Exception as e:
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32603, "message": str(e)}
        })

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        mcp.run()