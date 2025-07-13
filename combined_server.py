from typing import Any, Dict, List
import httpx
import os
import asyncio
from mcp.server.fastmcp import FastMCP
import json
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

# Initialize FastMCP server
mcp = FastMCP("azure-mcp-server")

# Wikibase Constants
WIKIBASE_BASE_URL = "https://mutaabiklegalresearch.wikibase.cloud"
SPARQL_ENDPOINT = f"{WIKIBASE_BASE_URL}/query/"
WIKIBASE_USER_AGENT = os.getenv("USER_AGENT", "WikibaseMCPTest/1.0")

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
                k=1
            )
        except Exception as e:
            print(f"Warning: Could not initialize GraphRAG system: {e}")
            self.embeddings = None
            self.example_selector = None

    async def generate_sparql_query(self, natural_language_input: str) -> str:
        """Select the most similar SPARQL query from examples."""
        try:
            if self.example_selector:
                selected_examples = self.example_selector.select_examples({"input": natural_language_input})
                if selected_examples:
                    return selected_examples[0]["query"]
            return SPARQL_EXAMPLES[0]["query"]
        except Exception as e:
            raise Exception(f"Failed to select SPARQL query: {str(e)}")


# Initialize the GraphRAG system
graph_rag = WikibaseGraphRAG()


async def execute_sparql_query(query: str) -> dict[str, Any] | None:
    """Execute a SPARQL query against the Wikibase instance."""
    headers = {
        "User-Agent": WIKIBASE_USER_AGENT,
        "Accept": "application/sparql-results+json"
    }
    data = {"query": query}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(SPARQL_ENDPOINT, data=data, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None


def format_sparql_results(data: dict) -> str:
    """Format SPARQL query results into a readable string."""
    if not data or "results" not in data or "bindings" not in data["results"]:
        return "No results found."

    bindings = data["results"]["bindings"]
    if not bindings:
        return "No results found."

    variables = list(bindings[0].keys())

    result_lines = []
    result_lines.append(" | ".join(variables))
    result_lines.append("-" * (len(" | ".join(variables)) + len(variables) * 3))

    for binding in bindings[:50]:
        row = []
        for var in variables:
            if var in binding:
                value = binding[var].get("value", "")
                if value.startswith("http://www.wikidata.org/entity/"):
                    value = value.split("/")[-1]
                elif value.startswith(WIKIBASE_BASE_URL):
                    value = value.split("/")[-1]
                if len(value) > 60:
                    value = value[:57] + "..."
                row.append(value)
            else:
                row.append("")
        result_lines.append(" | ".join(row))

    if len(bindings) > 50:
        result_lines.append(f"\n... and {len(bindings) - 50} more results")

    return "\n".join(result_lines)


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
        results = await execute_sparql_query(sparql_query)

        if not results:
            return f"Selected SPARQL query:\n{sparql_query}\n\nError: Unable to execute query or connection failed."

        # Format and return results
        formatted_results = format_sparql_results(results)

        return f"Selected SPARQL query:\n{sparql_query}\n\nResults:\n{formatted_results}"

    except Exception as e:
        return f"Error processing natural language query: {str(e)}"


if __name__ == "__main__":
    # Run as MCP stdio server (default for Claude.ai)
    mcp.run(transport='stdio')