# A stateful MCP server that holds a sympy sesssion, with symbol table of variables
# that can be used in the tools API to define and manipulate expressions.

from mcp.server.fastmcp import FastMCP
import sympy
import argparse
import logging
from typing import List, Dict, Optional
from pydantic import BaseModel
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.facts import InconsistentAssumptions
from vars import Assumption, Domain, ODEHint, PDEHint
from sympy import Eq, Function, dsolve
from sympy.solvers.pde import pdsolve

try:
    from einsteinpy.symbolic import RicciTensor, RicciScalar
    from einsteinpy.symbolic.predefined import AntiDeSitter
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Create an MCP server
mcp = FastMCP(
    "sympy-mcp",
    dependencies=["sympy", "pydantic"],
    instructions="Provides access to the Sympy computer algebra system, which can perform symbolic manipulation of mathematical expressions.",
)

# Global store for sympy variables and expressions
local_vars: Dict[str, sympy.Symbol] = {}
functions: Dict[str, sympy.Function] = {}
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
    # Merge local_vars and functions dictionaries to make both available for parsing
    parse_dict = {**local_vars, **functions}
    parsed_expr = parse_expr(expr_str, local_dict=parse_dict, evaluate=canonicalize)
    expr_key = f"expr_{expression_counter}"
    expressions[expr_key] = parsed_expr
    expression_counter += 1
    return expr_key


def introduce_equation(lhs_str: str, rhs_str: str) -> str:
    """Introduces an equation (lhs = rhs) using available local variables."""
    global expression_counter
    # Merge local_vars and functions dictionaries to make both available for parsing
    parse_dict = {**local_vars, **functions}
    lhs_expr = parse_expr(lhs_str, local_dict=parse_dict)
    rhs_expr = parse_expr(rhs_str, local_dict=parse_dict)
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
            # Get assumptions directly from the symbol object
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
def solve_algebraically(
    expr_key: str, solve_for_var_name: str, domain: Domain = Domain.COMPLEX
) -> str:
    """Solves an equation (expression = 0) algebraically for a given variable.

    Args:
        expr_key: The key of the expression (previously introduced) to be solved.
        solve_for_var_name: The name of the variable (previously introduced) to solve for.
        domain: The domain to solve in: Domain.COMPLEX, Domain.REAL, Domain.INTEGERS, or Domain.NATURALS. Defaults to Domain.COMPLEX.

    Returns:
        A LaTeX string representing the set of solutions. Returns an error message string if issues occur.
    """
    if expr_key not in expressions:
        return f"Error: Expression with key '{expr_key}' not found."

    expression_to_solve = expressions[expr_key]

    if solve_for_var_name not in local_vars:
        return f"Error: Variable '{solve_for_var_name}' not found in local_vars. Please introduce it first."

    variable_symbol = local_vars[solve_for_var_name]

    # Map domain enum to SymPy domain sets
    domain_map = {
        Domain.COMPLEX: sympy.S.Complexes,
        Domain.REAL: sympy.S.Reals,
        Domain.INTEGERS: sympy.S.Integers,
        Domain.NATURALS: sympy.S.Naturals0,
    }

    if domain not in domain_map:
        return "Error: Invalid domain. Choose from: Domain.COMPLEX, Domain.REAL, Domain.INTEGERS, or Domain.NATURALS."

    sympy_domain = domain_map[domain]

    try:
        # If the expression is an equation (Eq object), convert it to standard form
        if isinstance(expression_to_solve, sympy.Eq):
            expression_to_solve = expression_to_solve.lhs - expression_to_solve.rhs

        # Use solveset instead of solve
        solution_set = sympy.solveset(
            expression_to_solve, variable_symbol, domain=sympy_domain
        )

        # Convert the set to LaTeX format
        latex_output = sympy.latex(solution_set)
        return latex_output
    except NotImplementedError as e:
        return f"Error: SymPy could not solve the equation: {str(e)}. The equation may not have a closed-form solution or the algorithm is not implemented."
    except Exception as e:
        return f"An unexpected error occurred during solving: {str(e)}"


@mcp.tool()
def solve_linear_system(
    expr_keys: List[str], var_names: List[str], domain: Domain = Domain.COMPLEX
) -> str:
    """Solves a system of linear equations using SymPy's linsolve.

    Args:
        expr_keys: The keys of the expressions (previously introduced) forming the system.
        var_names: The names of the variables to solve for.
        domain: The domain to solve in (Domain.COMPLEX, Domain.REAL, etc.). Defaults to Domain.COMPLEX.

    Returns:
        A LaTeX string representing the solution set. Returns an error message string if issues occur.
    """
    # Validate all expression keys exist
    system = []
    for expr_key in expr_keys:
        if expr_key not in expressions:
            return f"Error: Expression with key '{expr_key}' not found."

        expr = expressions[expr_key]
        # Convert equations to standard form
        if isinstance(expr, sympy.Eq):
            expr = expr.lhs - expr.rhs
        system.append(expr)

    # Validate all variables exist
    symbols = []
    for var_name in var_names:
        if var_name not in local_vars:
            return f"Error: Variable '{var_name}' not found in local_vars. Please introduce it first."
        symbols.append(local_vars[var_name])

    # Map domain enum to SymPy domain sets
    domain_map = {
        Domain.COMPLEX: sympy.S.Complexes,
        Domain.REAL: sympy.S.Reals,
        Domain.INTEGERS: sympy.S.Integers,
        Domain.NATURALS: sympy.S.Naturals0,
    }

    if domain not in domain_map:
        return "Error: Invalid domain. Choose from: Domain.COMPLEX, Domain.REAL, Domain.INTEGERS, or Domain.NATURALS."

    domain_map[domain]

    try:
        # Use SymPy's linsolve - note: it doesn't take domain parameter directly, but works on the given domain
        solution_set = sympy.linsolve(system, symbols)

        # Convert the set to LaTeX format
        latex_output = sympy.latex(solution_set)
        return latex_output
    except NotImplementedError as e:
        return f"Error: SymPy could not solve the linear system: {str(e)}."
    except ValueError as e:
        return f"Error: Invalid system or arguments: {str(e)}."
    except Exception as e:
        return f"An unexpected error occurred during solving: {str(e)}"


@mcp.tool()
def solve_nonlinear_system(
    expr_keys: List[str], var_names: List[str], domain: Domain = Domain.COMPLEX
) -> str:
    """Solves a system of nonlinear equations using SymPy's nonlinsolve.

    Args:
        expr_keys: The keys of the expressions (previously introduced) forming the system.
        var_names: The names of the variables to solve for.
        domain: The domain to solve in (Domain.COMPLEX, Domain.REAL, etc.). Defaults to Domain.COMPLEX.

    Returns:
        A LaTeX string representing the solution set. Returns an error message string if issues occur.
    """
    # Validate all expression keys exist
    system = []
    for expr_key in expr_keys:
        if expr_key not in expressions:
            return f"Error: Expression with key '{expr_key}' not found."

        expr = expressions[expr_key]
        # Convert equations to standard form
        if isinstance(expr, sympy.Eq):
            expr = expr.lhs - expr.rhs
        system.append(expr)

    # Validate all variables exist
    symbols = []
    for var_name in var_names:
        if var_name not in local_vars:
            return f"Error: Variable '{var_name}' not found in local_vars. Please introduce it first."
        symbols.append(local_vars[var_name])

    # Map domain enum to SymPy domain sets
    domain_map = {
        Domain.COMPLEX: sympy.S.Complexes,
        Domain.REAL: sympy.S.Reals,
        Domain.INTEGERS: sympy.S.Integers,
        Domain.NATURALS: sympy.S.Naturals0,
    }

    if domain not in domain_map:
        return "Error: Invalid domain. Choose from: Domain.COMPLEX, Domain.REAL, Domain.INTEGERS, or Domain.NATURALS."

    try:
        # Use SymPy's nonlinsolve
        solution_set = sympy.nonlinsolve(system, symbols)

        # Convert the set to LaTeX format
        latex_output = sympy.latex(solution_set)
        return latex_output
    except NotImplementedError as e:
        return f"Error: SymPy could not solve the nonlinear system: {str(e)}."
    except ValueError as e:
        return f"Error: Invalid system or arguments: {str(e)}."
    except Exception as e:
        return f"An unexpected error occurred during solving: {str(e)}"


@mcp.tool()
def introduce_function(func_name: str) -> str:
    """Introduces a SymPy function variable and stores it.

    Takes a function name and creates a SymPy Function object for use in defining differential equations.

    Example:
        {func_name: "f"} will create the function f(x), f(t), etc. that can be used in expressions

    Returns:
        The name of the created function.
    """
    func = Function(func_name)
    functions[func_name] = func
    return func_name


@mcp.tool()
def dsolve_ode(expr_key: str, func_name: str, hint: Optional[ODEHint] = None) -> str:
    """Solves an ordinary differential equation using SymPy's dsolve function.

    Args:
        expr_key: The key of the expression (previously introduced) containing the differential equation.
        func_name: The name of the function (previously introduced) to solve for.
        hint: Optional solving method from ODEHint enum. If None, SymPy will try to determine the best method.

    Example:
        # First introduce a variable and a function
        intro("x", [Assumption.REAL], [])
        introduce_function("f")

        # Create a second-order ODE: f''(x) + 9*f(x) = 0
        expr_key = introduce_expression("Derivative(f(x), x, x) + 9*f(x)")

        # Solve the ODE
        result = dsolve_ode(expr_key, "f")
        # Returns solution with sin(3*x) and cos(3*x) terms

    Returns:
        A LaTeX string representing the solution. Returns an error message string if issues occur.
    """
    if expr_key not in expressions:
        return f"Error: Expression with key '{expr_key}' not found."

    if func_name not in functions:
        return f"Error: Function '{func_name}' not found. Please introduce it first using introduce_function."

    expression = expressions[expr_key]

    try:
        # Convert to equation form if it's not already
        if isinstance(expression, sympy.Eq):
            eq = expression
        else:
            eq = sympy.Eq(expression, 0)

        # Let SymPy handle function detection and apply the specified hint if provided
        if hint is not None:
            solution = dsolve(eq, hint=hint.value)
        else:
            solution = dsolve(eq)

        # Convert the solution to LaTeX format
        latex_output = sympy.latex(solution)
        return latex_output
    except ValueError as e:
        return f"Error: {str(e)}. This might be due to an invalid hint or an unsupported equation type."
    except NotImplementedError as e:
        return f"Error: Method not implemented: {str(e)}. Try a different hint or equation type."
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


@mcp.tool()
def pdsolve_pde(expr_key: str, func_name: str, hint: Optional[PDEHint] = None) -> str:
    """Solves a partial differential equation using SymPy's pdsolve function.

    Args:
        expr_key: The key of the expression (previously introduced) containing the PDE.
                 If the expression is not an equation (Eq), it will be interpreted as
                 PDE = 0.
        func_name: The name of the function (previously introduced) to solve for.
                   This should be a function of multiple variables.

    Example:
        # First introduce variables and a function
        intro("x", [Assumption.REAL], [])
        intro("y", [Assumption.REAL], [])
        introduce_function("f")

        # Create a PDE: 1 + 2*(ux/u) + 3*(uy/u) = 0
        expr_key = introduce_expression(
            "Eq(1 + 2*Derivative(f(x, y), x)/f(x, y) + 3*Derivative(f(x, y), y)/f(x, y), 0)"
        )

        # Solve the PDE
        result = pdsolve_pde(expr_key, "f")
        # Returns solution with exponential terms and arbitrary function

    Returns:
        A LaTeX string representing the solution. Returns an error message string if issues occur.
    """
    if expr_key not in expressions:
        return f"Error: Expression with key '{expr_key}' not found."

    if func_name not in functions:
        return f"Error: Function '{func_name}' not found. Please introduce it first using introduce_function."

    expression = expressions[expr_key]

    try:
        # Handle both equation and non-equation expressions
        if isinstance(expression, sympy.Eq):
            eq = expression
        else:
            eq = sympy.Eq(expression, 0)

        # Let SymPy's pdsolve find the dependent variable itself
        if hint is not None:
            solution = pdsolve(eq, hint=hint.value)
        else:
            solution = pdsolve(eq)

        # Convert the solution to LaTeX format
        latex_output = sympy.latex(solution)
        return latex_output
    except ValueError as e:
        return f"Error: {str(e)}. This might be due to an unsupported equation type."
    except NotImplementedError as e:
        return f"Error: Method not implemented: {str(e)}. The PDE might not be solvable using the implemented methods."
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


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
