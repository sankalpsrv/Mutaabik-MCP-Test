# Mutaabik MCP Server

A simplified Model Context Protocol (MCP) server for Wikibase SPARQL queries. Provides natural language to SPARQL query functionality for legal/environmental data.

## Features

- **Single Tool**: `natural_language_query(question)` - Query legal/environmental data using natural language
- **Simplified Architecture**: Single file, functional approach  
- **Dual Mode**: Works as both stdio MCP server and HTTP server
- **AI-Powered Query Selection**: Uses Azure OpenAI embeddings for semantic matching
- **Graceful Fallbacks**: Works without AI credentials (uses first example)
- **Wikibase Integration**: Uses wikibaseintegrator for robust SPARQL execution
- **Environment Variable Support**: Loads from .env files automatically

## Setup Instructions

### Prerequisites
- Python 3.11+
- UV package manager

### Installation

1. **Install UV (if not already installed):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

2. **Clone/Navigate to the project:**
```bash
cd azure-uv-mcp-server
```

3. **Install dependencies:**
```bash
uv sync
```

4. **Set environment variables:**

Option A: Create a `.env` file (recommended):
```bash
cp .env.example .env
# Edit .env file with your actual values
```

Option B: Export environment variables:
```bash
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_API_ENDPOINT="your-endpoint"
export USER_AGENT="MutaabikMCP/1.0"
```

## Running the Server

### Option 1: As MCP stdio Server (Default)
```bash
# Activate virtual environment
source .venv/bin/activate

# Run the server
python combined_server.py
```

### Option 2: As HTTP Server on Local Port
```bash
# Activate virtual environment
source .venv/bin/activate

# Run as HTTP server on port 8000
python combined_server.py server
```

The server will be available at `http://localhost:8000`

- Health check: `GET http://localhost:8000/health`
- Server info: `GET http://localhost:8000/`
- MCP SSE endpoint: `GET http://localhost:8000/message`
- MCP message handler: `POST http://localhost:8000/message`

## Testing the Server

### Test via MCP Inspector
1. Start the server in HTTP mode: `python combined_server.py server`
2. Open MCP Inspector in your browser
3. Connect to: `http://localhost:8000/message`
4. Test the `natural_language_query` tool

### Test via curl
```bash
# Test MCP message endpoint
curl -X POST http://localhost:8000/message \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "natural_language_query",
      "arguments": {
        "question": "Extract environmental law rule names"
      }
    }
  }'
```

## Environment Variables

- `AZURE_OPENAI_API_KEY`: Optional - enables AI-powered query selection
- `AZURE_OPENAI_API_ENDPOINT`: Optional - Azure OpenAI endpoint
- `USER_AGENT`: Optional, defaults to "MutaabikMCP/1.0"

## Architecture

The simplified architecture consists of:

1. **Functional Components**: Pure functions for query selection, execution, and formatting
2. **Graceful Initialization**: AI components initialize safely with fallbacks
3. **Single Entry Point**: One file with clear separation of concerns
4. **Minimal Dependencies**: Only essential packages required
5. **Environment Integration**: Automatic .env loading with fallbacks