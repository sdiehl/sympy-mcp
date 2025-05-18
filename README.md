# Sympy MCP Server

This is Model Context Protocol server that exposes stateful symbolic manipulation tools to a LLM. So it can manipulate mathematical expressions and equations.

## Why?

Language models absolutely *absolutely abysmal* at symbolic manipulation. They hallucinate variables, make up random constants, permute terms and generally make a mess. But we have computer algebra systems specifically for symbolic manipulation, so we can use tool-calling to orchestrate a sequence of transforms so that the CAS does all the heavy lifting.

While you can certainly have an LLM generate Mathematica or Python code, if you want to use the LLM as on-the-fly calculator, it's a better user experience to use the MCP server and expose the symbolic tools directly.

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

## Docker Usage

You can also run the server using Docker:

```bash
# Build the Docker image
docker build -t sympy-mcp .

# Run the Docker container
docker run -p 8081:8081 sympy-mcp
```

### Claude Desktop Configuration

To configure Claude Desktop to launch the Docker container, edit your `claude_desktop_config.json` file (usually located at `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "sympy-mcp": {
      "command": "docker",
      "args": [
        "run",
        "--rm",
        "-p",
        "8081:8081",
        "sympy-mcp"
      ]
    }
  }
}
```

This configuration tells Claude Desktop to launch the Docker container when needed. Make sure to build the Docker image (`docker build -t sympy-mcp .`) before using Claude Desktop with this configuration.

## Security Disclaimer

This server runs on your computer and gives Claude access to run Python logic. Notably it uses Sympy's `parse_expr` to parse mathematical expressions, which is uses `eval` under the hood, effectively allowing arbitrary code execution. By running the server, you are trusting the code that Claude generates. Running in the Docker image is slightly safer, but it's still a good idea to review the code before running it.