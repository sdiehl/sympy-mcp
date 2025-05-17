# A stateful MCP server that holds a sympy sesssion, with symbol table of variables
# that can be used in the tools API to define and manipulate expressions.

from mcp.server.fastmcp import FastMCP
import sympy
import argparse
import logging
from typing import List, Dict
from pydantic import BaseModel  # Added for Pydantic model
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.facts import InconsistentAssumptions
from vars import Assumption
from sympy import Eq, FiniteSet  # Added for creating solution sets

logger = logging.getLogger(__name__)

# Create an MCP server
mcp = FastMCP("sympy-mcp", dependencies=["sympy", "pydantic"])

# Global store for sympy variables and expressions
local_vars: Dict[str, sympy.Symbol] = {}
expressions: Dict[str, sympy.Expr] = {}
expression_counter = 0


# Pydantic model for defining a variable with assumptions
class VariableDefinition(BaseModel):
    var_name: str
    pos_assumptions: List[str] = []
    neg_assumptions: List[str] = []


# x, y = symbols('x, y', commutative=False)


# Add an addition tool
@mcp.tool()
def intro(
    var_name: str, pos_assumptions: List[Assumption], neg_assumptions: List[Assumption]
) -> str:
    """Introduces a sympy variable with specified assumptions and stores it.

    Takes a variable name and a list of positive and negative assumptions.
    """
    kwargs_for_symbols = {}
    # Add assumptions
    for assumption_obj in pos_assumptions:
        kwargs_for_symbols[assumption_obj.value] = True

    for assumption_obj in neg_assumptions:
        kwargs_for_symbols[assumption_obj.value] = False

    try:
        var = sympy.symbols(var_name, **kwargs_for_symbols)
    except InconsistentAssumptions as e:
        return f"Error creating symbol '{var_name}': The provided assumptions {kwargs_for_symbols} are inconsistent according to SymPy. Details: {str(e)}"
    except Exception as e:
        return f"Error creating symbol '{var_name}': An unexpected error occurred. Assumptions attempted: {kwargs_for_symbols}. Details: {type(e).__name__} - {str(e)}"

    local_vars[var_name] = var
    return var_name


# Introduce multiple variables simultaneously
@mcp.tool()
def intro_many(variables: List[VariableDefinition]) -> str:
    """Introduces multiple sympy variables with specified assumptions and stores them.

    Takes a list of VariableDefinition objects for the 'variables' parameter.
    Each object in the list specifies:
    - var_name: The name of the variable (string).
    - pos_assumptions: A list of positive assumption strings (e.g., ["real", "positive"]).
    - neg_assumptions: A list of negative assumption strings (e.g., ["complex"]).

    The JSON payload for the 'variables' argument should be a direct list of these objects, for example:
    ```json
    [
        {
            "var_name": "x",
            "pos_assumptions": ["real", "positive"],
            "neg_assumptions": ["complex"]
        },
        {
            "var_name": "y",
            "pos_assumptions": [],
            "neg_assumptions": ["commutative"]
        }
    ]
    ```

    The assumptions must be consistent, so a real number is not allowed to be non-commutative.

    Prefer this over intro() for multiple variables because it's more efficient.
    """
    var_keys = {}
    for var_def in variables:
        try:
            processed_pos_assumptions = [
                Assumption(a_str) for a_str in var_def.pos_assumptions
            ]
            processed_neg_assumptions = [
                Assumption(a_str) for a_str in var_def.neg_assumptions
            ]
        except ValueError as e:
            # Handle cases where a string doesn't match an Assumption enum member
            msg = (
                f"Error for variable '{var_def.var_name}': Invalid assumption string provided. {e}. "
                f"Ensure assumptions match valid enum values in 'vars.Assumption'."
            )
            logger.error(msg)
            return msg  # Or collect errors

        var_key = intro(
            var_def.var_name, processed_pos_assumptions, processed_neg_assumptions
        )
        var_keys[var_def.var_name] = var_key

    # Return the mapping of variable names to keys
    return str(var_keys)


# XXX use local_vars {x : "expr_1", y : "expr_2"}
@mcp.tool()
def introduce_expression(expr_str: str, canonicalize: bool = True) -> str:
    """Parses a sympy expression string using available local variables and stores it.

    Uses Sympy parse_expr to parse the expression string.

    Applies default Sympy canonicalization rules unless canonicalize is False.

    For equations (x^2 = 1) make the input string "Eq(x^2, 1") not "x^2 == 1"

    Examples:

        {expr_str: "Eq(x^2 + y^2, 1)"}
        {expr_str: "Matrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11)))"}
    """
    global expression_counter
    parsed_expr = parse_expr(expr_str, local_dict=local_vars, evaluate=canonicalize)
    expr_key = f"expr_{expression_counter}"
    expressions[expr_key] = parsed_expr
    expression_counter += 1
    return expr_key


def introduce_equation(lhs_str: str, rhs_str: str) -> str:
    """Introduces an equation (lhs = rhs) using available local variables."""
    global expression_counter
    lhs_expr = parse_expr(lhs_str, local_dict=local_vars)
    rhs_expr = parse_expr(rhs_str, local_dict=local_vars)
    eq_key = f"eq_{expression_counter}"
    expressions[eq_key] = Eq(lhs_expr, rhs_expr)
    expression_counter += 1
    return eq_key


@mcp.tool()
def print_latex_expression(expr_key: str) -> str:
    """Prints a stored expression in LaTeX format, along with variable assumptions."""
    if expr_key not in expressions:
        return f"Error: Expression key '{expr_key}' not found."

    expr = expressions[expr_key]
    latex_str = sympy.latex(expr)

    # Find variables in the expression and their assumptions
    variables_in_expr = expr.free_symbols
    assumption_descs = []
    for var_symbol in variables_in_expr:
        var_name = str(var_symbol)
        if var_name in local_vars:
            # Accessing internal _assumptions dictionary; might need a more robust way if API changes
            # For now, sympy.Symbol().assumptions0 gives a good summary string
            # however, assumptions_dict used in introduce is more precise.
            # We reconstruct the assumption string from what we stored.
            # Getting assumptions directly from the symbol object can be complex.
            # Let's find the original assumptions used during creation.
            # This requires a way to store/retrieve original assumptions with the symbol if not directly on it.
            # For now, we'll just state the variable is defined.
            # A better approach would be to store assumption list alongside the symbol in local_vars
            # or parse them from symbol.assumptions0 if it's reliable.

            # Let's try to get assumptions directly from the symbol object
            current_assumptions = []
            # sympy stores assumptions in a private attribute _assumptions
            # and provides a way to query them via .is_commutative, .is_real etc.
            # We can iterate through known Assumption enum values
            for assumption_enum_member in Assumption:
                if getattr(var_symbol, f"is_{assumption_enum_member.value}", False):
                    current_assumptions.append(assumption_enum_member.value)

            if current_assumptions:
                assumption_descs.append(
                    f"{var_name} is {', '.join(current_assumptions)}"
                )
            else:
                assumption_descs.append(f"{var_name} (no specific assumptions listed)")
        else:
            assumption_descs.append(f"{var_name} (undefined in local_vars)")

    if assumption_descs:
        return f"{latex_str} (where {'; '.join(assumption_descs)})"
    else:
        return latex_str


@mcp.tool()
def solve_algebraically(expr_key: str, solve_for_var_name: str) -> str:
    """Solves an equation (expression = 0) algebraically for a given variable.

    Args:
        expr_key: The key of the expression (previously introduced) to be solved.
        solve_for_var_name: The name of the variable (previously introduced) to solve for.

    Returns:
        A LaTeX string representing the set of solutions. Returns an error message string if issues occur.
    """
    if expr_key not in expressions:
        return f"Error: Expression with key '{expr_key}' not found."

    expression_to_solve = expressions[expr_key]

    if solve_for_var_name not in local_vars:
        return f"Error: Variable '{solve_for_var_name}' not found in local_vars. Please introduce it first."

    variable_symbol = local_vars[solve_for_var_name]

    try:
        # sympy.solve assumes the expression is equal to 0 and for a single variable, returns a list of solutions.
        solutions_list = sympy.solve(expression_to_solve, variable_symbol)

        # Create a FiniteSet from the list of solutions
        # If solutions_list is empty, FiniteSet will correctly create an EmptySet.
        solution_set = FiniteSet(*solutions_list)

        # Convert the set to LaTeX format
        latex_output = sympy.latex(solution_set)
        return latex_output
    except NotImplementedError as e:
        return f"Error: SymPy could not solve the equation: {str(e)}. The equation may not have a closed-form solution or the algorithm is not implemented."
    except Exception as e:
        return f"An unexpected error occurred during solving: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="MCP server for SymPy")
    parser.add_argument(
        "--mcp-host",
        type=str,
        default="127.0.0.1",
        help="Host to run MCP server on (only used for sse), default: 127.0.0.1",
    )
    parser.add_argument(
        "--mcp-port",
        type=int,
        help="Port to run MCP server on (only used for sse), default: 8081",
    )
    parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        choices=["stdio", "sse"],
        help="Transport protocol for MCP, default: stdio",
    )
    args = parser.parse_args()

    if args.transport == "sse":
        try:
            # Set up logging
            log_level = logging.INFO
            logging.basicConfig(level=log_level)
            logging.getLogger().setLevel(log_level)

            # Configure MCP settings
            mcp.settings.log_level = "INFO"
            if args.mcp_host:
                mcp.settings.host = args.mcp_host
            else:
                mcp.settings.host = "127.0.0.1"

            if args.mcp_port:
                mcp.settings.port = args.mcp_port
            else:
                mcp.settings.port = 8081

            logger.info(
                f"Starting MCP server on http://{mcp.settings.host}:{mcp.settings.port}/sse"
            )
            logger.info(f"Using transport: {args.transport}")

            mcp.run(transport="sse")
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
    else:
        print("Starting MCP server with stdio transport")
        mcp.run()


if __name__ == "__main__":
    main()
