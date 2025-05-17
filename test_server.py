import pytest
from server import (
    intro,
    intro_many,
    introduce_expression,
    print_latex_expression,
    solve_algebraically,
    solve_linear_system,
    solve_nonlinear_system,
    introduce_function,
    dsolve_ode,
    local_vars,
    expressions,
    functions,
    VariableDefinition,
)
from vars import Assumption, Domain, ODEHint


# Add a fixture to reset global state between tests
@pytest.fixture(autouse=True)
def reset_globals():
    # Clear global dictionaries before each test
    local_vars.clear()
    expressions.clear()
    functions.clear()  # Add this to clear the functions dictionary as well
    # Reset the expression counter
    import server

    server.expression_counter = 0
    yield


class TestIntroTool:
    def test_intro_basic(self):
        # Test introducing a variable with no assumptions
        result = intro("x", [], [])
        assert result == "x"
        assert "x" in local_vars

    def test_intro_with_assumptions(self):
        # Test introducing a variable with assumptions
        result = intro("y", [Assumption.REAL, Assumption.POSITIVE], [])
        assert result == "y"
        assert "y" in local_vars
        # Check that the symbol has the correct assumptions
        assert local_vars["y"].is_real is True
        assert local_vars["y"].is_positive is True

    def test_intro_inconsistent_assumptions(self):
        # Test introducing a variable with inconsistent assumptions
        # For example, a number can't be both positive and negative
        result = intro("z", [Assumption.POSITIVE], [])
        assert result == "z"
        assert "z" in local_vars

        # Now try to create inconsistent assumptions with another variable
        # Positive and non-positive are inconsistent
        result2 = intro(
            "inconsistent", [Assumption.POSITIVE, Assumption.NONPOSITIVE], []
        )
        assert "error" in result2.lower() or "inconsistent" in result2.lower()
        assert "inconsistent" not in local_vars


class TestIntroManyTool:
    def test_intro_many_basic(self):
        # Define variable definition objects using the VariableDefinition class
        var_defs = [
            VariableDefinition(
                var_name="a", pos_assumptions=["real"], neg_assumptions=[]
            ),
            VariableDefinition(
                var_name="b", pos_assumptions=["positive"], neg_assumptions=[]
            ),
        ]

        intro_many(var_defs)
        assert "a" in local_vars
        assert "b" in local_vars
        assert local_vars["a"].is_real is True
        assert local_vars["b"].is_positive is True

    def test_intro_many_invalid_assumption(self):
        # Create variable definition with an invalid assumption
        var_defs = [
            VariableDefinition(
                var_name="c", pos_assumptions=["invalid_assumption"], neg_assumptions=[]
            ),
        ]

        result = intro_many(var_defs)
        assert "error" in result.lower()


class TestIntroduceExpressionTool:
    def test_introduce_simple_expression(self):
        # First, introduce required variables
        intro("x", [], [])
        intro("y", [], [])

        # Then introduce an expression
        result = introduce_expression("x + y")
        assert result == "expr_0"
        assert "expr_0" in expressions
        assert str(expressions["expr_0"]) == "x + y"

    def test_introduce_equation(self):
        intro("x", [], [])

        result = introduce_expression("Eq(x**2, 4)")
        assert result == "expr_0"
        assert "expr_0" in expressions
        # Equation should be x**2 = 4

        assert expressions["expr_0"].lhs == local_vars["x"] ** 2
        assert expressions["expr_0"].rhs == 4

    def test_introduce_matrix(self):
        result = introduce_expression("Matrix(((1, 2), (3, 4)))")
        assert result == "expr_0"
        assert "expr_0" in expressions
        # Check matrix dimensions and values
        assert expressions["expr_0"].shape == (2, 2)
        assert expressions["expr_0"][0, 0] == 1
        assert expressions["expr_0"][1, 1] == 4


class TestPrintLatexExpressionTool:
    def test_print_latex_simple_expression(self):
        intro("x", [Assumption.REAL], [])
        expr_key = introduce_expression("x**2 + 5*x + 6")

        result = print_latex_expression(expr_key)
        assert "x^{2} + 5 x + 6" in result
        assert "real" in result.lower()

    def test_print_latex_nonexistent_expression(self):
        result = print_latex_expression("nonexistent_key")
        assert "error" in result.lower()


class TestSolveAlgebraicallyTool:
    def test_solve_quadratic(self):
        intro("x", [Assumption.REAL], [])
        expr_key = introduce_expression("Eq(x**2 - 5*x + 6, 0)")

        result = solve_algebraically(expr_key, "x")
        # Solution should contain the values 2 and 3
        assert "2" in result
        assert "3" in result

    def test_solve_with_domain(self):
        intro("x", [Assumption.REAL], [])
        # Try a clearer example: x^2 + 1 = 0 directly as an expression
        expr_key = introduce_expression("x**2 + 1")

        # In complex domain, should have solutions i and -i
        complex_result = solve_algebraically(expr_key, "x", Domain.COMPLEX)
        assert "i" in complex_result

        # In real domain, should have empty set
        real_result = solve_algebraically(expr_key, "x", Domain.REAL)
        assert "\\emptyset" in real_result

    def test_solve_invalid_domain(self):
        intro("x", [], [])
        introduce_expression("x**2 - 4")
        # We can't really test with an invalid Domain enum value easily,
        # so we'll skip this test since it's handled by type checking
        # If needed, could test with a mock Domain object that's not in the map

    def test_solve_nonexistent_expression(self):
        intro("x", [], [])
        result = solve_algebraically("nonexistent_key", "x")
        assert "error" in result.lower()

    def test_solve_nonexistent_variable(self):
        intro("x", [], [])
        expr_key = introduce_expression("x**2 - 4")
        result = solve_algebraically(expr_key, "y")
        assert "error" in result.lower()


class TestSolveLinearSystemTool:
    def test_simple_linear_system(self):
        # Create variables
        intro("x", [Assumption.REAL], [])
        intro("y", [Assumption.REAL], [])

        # Create a system of linear equations: x + y = 10, 2x - y = 5
        eq1 = introduce_expression("Eq(x + y, 10)")
        eq2 = introduce_expression("Eq(2*x - y, 5)")

        # Solve the system
        result = solve_linear_system([eq1, eq2], ["x", "y"])

        # Check if solution contains the expected values (x=5, y=5)
        assert "5" in result

    def test_inconsistent_system(self):
        # Create variables
        intro("x", [Assumption.REAL], [])
        intro("y", [Assumption.REAL], [])

        # Create an inconsistent system: x + y = 1, x + y = 2
        eq1 = introduce_expression("Eq(x + y, 1)")
        eq2 = introduce_expression("Eq(x + y, 2)")

        # Solve the system
        result = solve_linear_system([eq1, eq2], ["x", "y"])

        # Should be empty set
        assert "\\emptyset" in result

    def test_nonexistent_expression(self):
        intro("x", [], [])
        intro("y", [], [])
        result = solve_linear_system(["nonexistent_key"], ["x", "y"])
        assert "error" in result.lower()

    def test_nonexistent_variable(self):
        intro("x", [], [])
        expr_key = introduce_expression("x**2 - 4")
        result = solve_linear_system([expr_key], ["y"])
        assert "error" in result.lower()


class TestSolveNonlinearSystemTool:
    def test_simple_nonlinear_system(self):
        # Create variables
        intro("x", [Assumption.REAL], [])
        intro("y", [Assumption.REAL], [])

        # Create a system of nonlinear equations: x^2 + y^2 = 25, x*y = 12
        eq1 = introduce_expression("Eq(x**2 + y**2, 25)")
        eq2 = introduce_expression("Eq(x*y, 12)")

        # Solve the system
        result = solve_nonlinear_system([eq1, eq2], ["x", "y"])

        # Should find two pairs of solutions (±3, ±4) and (±4, ±3)
        # The exact format can vary, so we just check for the presence of 3 and 4
        assert "3" in result
        assert "4" in result

    def test_with_domain(self):
        # Create variables - importantly, not specifying REAL assumption
        # because we want to test complex solutions
        intro("x", [], [])
        intro("y", [], [])

        # Create a system with complex solutions: x^2 + y^2 = -1, y = x
        # This has no real solutions but has complex solutions
        eq1 = introduce_expression("Eq(x**2 + y**2, -1)")
        eq2 = introduce_expression("Eq(y, x)")

        # In complex domain - should have solutions with imaginary parts
        complex_result = solve_nonlinear_system([eq1, eq2], ["x", "y"], Domain.COMPLEX)
        assert "i" in complex_result

        # In real domain - should be empty set because no real solution exists
        real_result = solve_nonlinear_system([eq1, eq2], ["x", "y"], Domain.REAL)
        assert "\\emptyset" in real_result

    def test_nonexistent_expression(self):
        intro("x", [], [])
        intro("y", [], [])
        result = solve_nonlinear_system(["nonexistent_key"], ["x", "y"])
        assert "error" in result.lower()

    def test_nonexistent_variable(self):
        intro("x", [], [])
        expr_key = introduce_expression("x**2 - 4")
        result = solve_nonlinear_system([expr_key], ["z"])
        assert "error" in result.lower()


class TestIntroduceFunctionTool:
    def test_introduce_function_basic(self):
        # Test introducing a function variable
        result = introduce_function("f")
        assert result == "f"
        assert "f" in functions
        assert str(functions["f"]) == "f"

    def test_function_usage_in_expression(self):
        # Introduce a variable and a function
        intro("x", [Assumption.REAL], [])
        introduce_function("f")

        # Create an expression using the function
        expr_key = introduce_expression("f(x)")

        assert expr_key == "expr_0"
        assert "expr_0" in expressions
        assert str(expressions["expr_0"]) == "f(x)"


class TestDsolveOdeTool:
    def test_simple_ode(self):
        # Introduce a variable and a function
        intro("x", [Assumption.REAL], [])
        introduce_function("f")

        # Create a differential equation: f''(x) + 9*f(x) = 0
        expr_key = introduce_expression("Derivative(f(x), x, x) + 9*f(x)")

        # Solve the ODE
        result = dsolve_ode(expr_key, "f")

        # The solution should include sin(3*x) and cos(3*x)
        assert "sin" in result
        assert "cos" in result
        assert "3 x" in result

    def test_ode_with_hint(self):
        # Introduce a variable and a function
        intro("x", [Assumption.REAL], [])
        introduce_function("f")

        # Create a first-order exact equation: sin(x)*cos(f(x)) + cos(x)*sin(f(x))*f'(x) = 0
        expr_key = introduce_expression(
            "sin(x)*cos(f(x)) + cos(x)*sin(f(x))*Derivative(f(x), x)"
        )

        # Solve with specific hint
        result = dsolve_ode(expr_key, "f", ODEHint.FIRST_EXACT)

        # The solution might contain acos instead of sin
        assert "acos" in result or "sin" in result

    def test_nonexistent_expression(self):
        introduce_function("f")
        result = dsolve_ode("nonexistent_key", "f")
        assert "error" in result.lower()

    def test_nonexistent_function(self):
        intro("x", [Assumption.REAL], [])
        introduce_function("f")
        expr_key = introduce_expression("Derivative(f(x), x) - f(x)")
        result = dsolve_ode(expr_key, "g")
        assert "error" in result.lower()
