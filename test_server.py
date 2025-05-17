import pytest
from server import (
    intro,
    intro_many,
    introduce_expression,
    print_latex_expression,
    solve_algebraically,
    local_vars,
    expressions,
    VariableDefinition,
)
from vars import Assumption


# Add a fixture to reset global state between tests
@pytest.fixture(autouse=True)
def reset_globals():
    # Clear global dictionaries before each test
    local_vars.clear()
    expressions.clear()
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

    def test_solve_nonexistent_expression(self):
        intro("x", [], [])
        result = solve_algebraically("nonexistent_key", "x")
        assert "error" in result.lower()

    def test_solve_nonexistent_variable(self):
        intro("x", [], [])
        expr_key = introduce_expression("x**2 - 4")
        result = solve_algebraically(expr_key, "y")
        assert "error" in result.lower()
