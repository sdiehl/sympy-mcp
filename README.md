# Sympy MCP Server

This is Model Context Protocol server that exposes stateful symbolic manipulation tools to a LLM. So it can manipulate mathematical expressions and equations.

## Usage

```bash
uv sync
uv run mcp install server.py
uv run mcp run server.py
```

For development, you can run the server in watch mode:

```bash
uv run mcp dev server.py
```