import pytest
from server import (
    intro,
    introduce_expression,
    differentiate_expression,
    integrate_expression,
    create_coordinate_system,
    create_vector_field,
    calculate_curl,
    calculate_divergence,
    calculate_gradient,
    print_latex_expression,
    local_vars,
    expressions,
    coordinate_systems,
)
from vars import Assumption


# Add a fixture to reset global state between tests
@pytest.fixture(autouse=True)
def reset_globals():
    # Clear global dictionaries before each test
    local_vars.clear()
    expressions.clear()
    coordinate_systems.clear()
    # Reset the expression counter
    import server

    server.expression_counter = 0
    yield


class TestDifferentiateExpressionTool:
    def test_differentiate_polynomial(self):
        # Introduce a variable
        intro("x", [Assumption.REAL], [])

        # Create an expression: x^3
        expr_key = introduce_expression("x**3")

        # First derivative
        first_deriv_key = differentiate_expression(expr_key, "x")
        first_deriv_latex = print_latex_expression(first_deriv_key)

        # Should be 3x^2
        assert "3" in first_deriv_latex
        assert "x^{2}" in first_deriv_latex

        # Second derivative
        second_deriv_key = differentiate_expression(expr_key, "x", 2)
        second_deriv_latex = print_latex_expression(second_deriv_key)

        # Should be 6x
        assert "6" in second_deriv_latex
        assert "x" in second_deriv_latex

        # Third derivative
        third_deriv_key = differentiate_expression(expr_key, "x", 3)
        third_deriv_latex = print_latex_expression(third_deriv_key)

        # Should be 6
        assert "6" in third_deriv_latex

    def test_differentiate_trigonometric(self):
        # Introduce a variable
        intro("x", [Assumption.REAL], [])

        # Create sin(x) expression
        sin_key = introduce_expression("sin(x)")

        # Derivative of sin(x) is cos(x)
        deriv_key = differentiate_expression(sin_key, "x")
        deriv_latex = print_latex_expression(deriv_key)

        assert "\\cos" in deriv_latex

    def test_nonexistent_expression(self):
        intro("x", [Assumption.REAL], [])
        result = differentiate_expression("nonexistent_key", "x")
        assert "error" in result.lower()

    def test_nonexistent_variable(self):
        intro("x", [Assumption.REAL], [])
        expr_key = introduce_expression("x**2")
        result = differentiate_expression(expr_key, "y")
        assert "error" in result.lower()


class TestIntegrateExpressionTool:
    def test_indefinite_integral_polynomial(self):
        # Introduce a variable
        intro("x", [Assumption.REAL], [])

        # Create expression: x^2
        expr_key = introduce_expression("x**2")

        # Integrate
        integral_key = integrate_expression(expr_key, "x")
        integral_latex = print_latex_expression(integral_key)

        # Should be x^3/3
        assert "x^{3}" in integral_latex
        assert "3" in integral_latex

    def test_indefinite_integral_trigonometric(self):
        # Introduce a variable
        intro("x", [Assumption.REAL], [])

        # Create expression: cos(x)
        expr_key = introduce_expression("cos(x)")

        # Integrate
        integral_key = integrate_expression(expr_key, "x")
        integral_latex = print_latex_expression(integral_key)

        # Should be sin(x)
        assert "\\sin" in integral_latex

    def test_nonexistent_expression(self):
        intro("x", [Assumption.REAL], [])
        result = integrate_expression("nonexistent_key", "x")
        assert "error" in result.lower()

    def test_nonexistent_variable(self):
        intro("x", [Assumption.REAL], [])
        expr_key = introduce_expression("x**2")
        result = integrate_expression(expr_key, "y")
        assert "error" in result.lower()


class TestVectorOperations:
    def test_create_coordinate_system(self):
        # Create coordinate system
        result = create_coordinate_system("R")
        assert result == "R"
        assert "R" in coordinate_systems

    def test_create_custom_coordinate_system(self):
        # Create coordinate system with custom names
        result = create_coordinate_system("C", ["rho", "phi", "z"])
        assert result == "C"
        assert "C" in coordinate_systems

    def test_create_vector_field(self):
        # Create coordinate system
        create_coordinate_system("R")

        # Introduce variables to represent components
        intro("x", [Assumption.REAL], [])
        intro("y", [Assumption.REAL], [])
        intro("z", [Assumption.REAL], [])

        # Create vector field F = (y, -x, z)
        vector_field_key = create_vector_field("R", "y", "-x", "z")

        # The key might be an error message if the test is failing
        if "error" not in vector_field_key.lower():
            assert vector_field_key.startswith("vector_")
        else:
            assert False, f"Failed to create vector field: {vector_field_key}"

    def test_calculate_curl(self):
        # Create coordinate system
        create_coordinate_system("R")

        # Introduce variables
        intro("x", [Assumption.REAL], [])
        intro("y", [Assumption.REAL], [])

        # Create a simple vector field for curl calculation
        vector_field_key = create_vector_field("R", "y", "-x", "0")

        # Check if vector field was created successfully
        if "error" in vector_field_key.lower():
            assert False, f"Failed to create vector field: {vector_field_key}"

        # Calculate curl
        curl_key = calculate_curl(vector_field_key)

        # Check if curl calculation was successful
        if "error" not in curl_key.lower():
            assert curl_key.startswith("vector_")
        else:
            assert False, f"Failed to calculate curl: {curl_key}"

    def test_calculate_divergence(self):
        # Create coordinate system
        create_coordinate_system("R")

        # Introduce variables
        intro("x", [Assumption.REAL], [])
        intro("y", [Assumption.REAL], [])
        intro("z", [Assumption.REAL], [])

        # Create a simple identity vector field
        vector_field_key = create_vector_field("R", "x", "y", "z")

        # Check if vector field was created successfully
        if "error" in vector_field_key.lower():
            assert False, f"Failed to create vector field: {vector_field_key}"

        # Calculate divergence - should be 0 because symbols have no dependency on coordinates
        div_key = calculate_divergence(vector_field_key)

        # Check if divergence calculation was successful
        if "error" in div_key.lower():
            assert False, f"Failed to calculate divergence: {div_key}"

        div_latex = print_latex_expression(div_key)

        # Check result - should be 0
        assert "0" in div_latex

    def test_calculate_gradient(self):
        # Create coordinate system
        create_coordinate_system("R")

        # Introduce variables
        intro("x", [Assumption.REAL], [])
        intro("y", [Assumption.REAL], [])
        intro("z", [Assumption.REAL], [])

        # Create a simple scalar field
        scalar_field_key = introduce_expression("x**2 + y**2 + z**2")

        # Calculate gradient
        grad_key = calculate_gradient(scalar_field_key)

        # Check if gradient calculation was successful
        if "error" not in grad_key.lower():
            assert grad_key.startswith("vector_")
        else:
            assert False, f"Failed to calculate gradient: {grad_key}"

    def test_nonexistent_coordinate_system(self):
        result = create_vector_field("NonExistent", "x", "y", "z")
        assert "error" in result.lower()

    def test_nonexistent_vector_field(self):
        result = calculate_curl("nonexistent_key")
        assert "error" in result.lower()
