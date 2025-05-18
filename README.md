<div align="center">
  <img src=".github/logo.png" alt="Sympy MCP Logo" width="400" />
</div>

# Symbolic Algebra MCP Server

This is Model Context Protocol server that exposes stateful symbolic manipulation tools to a LLM. So it can manipulate mathematical expressions and equations.

## Why?

Language models are *absolutely abysmal* at symbolic manipulation. They hallucinate variables, make up random constants, permute terms and generally make a mess. But we have computer algebra systems specifically for symbolic manipulation, so we can use tool-calling to orchestrate a sequence of transforms so that the CAS does all the heavy lifting.

While you can certainly have an LLM generate Mathematica or Python code, if you want to use the LLM as on-the-fly calculator, it's a better user experience to use the MCP server and expose the symbolic tools directly.

## Available Tools

The sympy-mcp server provides the following tools for symbolic mathematics:

- **intro** - Introduces a variable with specified assumptions and stores it
- **intro_many** - Introduces multiple variables with specified assumptions simultaneously
- **introduce_expression** - Parses an expression string using available local variables and stores it
- **print_latex_expression** - Prints a stored expression in LaTeX format, along with variable assumptions
- **solve_algebraically** - Solves an equation algebraically for a given variable
- **solve_linear_system** - Solves a system of linear equations
- **solve_nonlinear_system** - Solves a system of nonlinear equations
- **introduce_function** - Introduces a function variable for use in differential equations
- **dsolve_ode** - Solves an ordinary differential equation
- **pdsolve_pde** - Solves a partial differential equation
- **create_predefined_metric** - Creates a predefined spacetime metric (e.g. Schwarzschild, Kerr, Minkowski)
- **search_predefined_metrics** - Searches available predefined metrics
- **calculate_tensor** - Calculates tensors from a metric (Ricci, Einstein, Weyl tensors)
- **create_custom_metric** - Creates a custom metric tensor from provided components and symbols
- **print_latex_tensor** - Prints a stored tensor expression in LaTeX format
- **simplify_expression** - Simplifies a mathematical expression
- **integrate_expression** - Integrates an expression with respect to a variable
- **differentiate_expression** - Differentiates an expression with respect to a variable
- **create_coordinate_system** - Creates a 3D coordinate system for vector calculus operations
- **create_vector_field** - Creates a vector field in the specified coordinate system
- **calculate_curl** - Calculates the curl of a vector field
- **calculate_divergence** - Calculates the divergence of a vector field
- **calculate_gradient** - Calculates the gradient of a scalar field

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

## Cursor Installation

In your ~/.cursor/mcp.json, add the following, where `REPLACE_WITH_PATH_TO_SYMPY_MCP` is the path to the sympy-mcp server.py file.

```json
{
  "mcpServers": {
    "sympy-mcp": {
      "command": "/opt/homebrew/bin/uv",
      "args": [
        "run",
        "--with",
        "einsteinpy",
        "--with",
        "mcp[cli]",
        "--with",
        "pydantic",
        "--with",
        "sympy",
        "mcp",
        "run",
        "REPLACE_WITH_PATH_TO_SYMPY_MCP/server.py"
      ]
    }
  }
}
```

## Cline Integration

To use with [Cline](https://cline.bot/), you need to manually run the MCP server first using the commands in the "Usage" section. Once the MCP server is running, open Cline and select "MCP Servers" at the top.

Then select "Remote Servers" and add the following:

- Server Name: sympy-mcp
- Server URL: http://127.0.0.1:8081/sse

## 5ire Integration

To set up with 5ire, open [5ire](https://github.com/nanbingxyz/5ire) and go to Tools -> New and set the following configurations:

- Tool Key: sympy-mcp
- Name: SymPy MCP
- Command: /opt/homebrew/bin/uv run --with einsteinpy --with mcp[cli] --with pydantic --with sympy mcp run /ABSOLUTE_PATH_TO/server.py

Replace `/ABSOLUTE_PATH_TO/server.py` with the actual path to your sympy-mcp server.py file.

## Security Disclaimer

This server runs on your computer and gives Claude access to run Python logic. Notably it uses Sympy's `parse_expr` to parse mathematical expressions, which is uses `eval` under the hood, effectively allowing arbitrary code execution. By running the server, you are trusting the code that Claude generates. Running in the Docker image is slightly safer, but it's still a good idea to review the code before running it.