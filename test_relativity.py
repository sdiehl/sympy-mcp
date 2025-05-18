import pytest
from server import (
    create_predefined_metric,
    search_predefined_metrics,
    calculate_tensor,
    create_custom_metric,
    print_latex_tensor,
    local_vars,
    expressions,
    metrics,
    tensor_objects,
    EINSTEINPY_AVAILABLE,
)
from vars import Metric, Tensor


# Skip all tests if EinsteinPy is not available
pytestmark = pytest.mark.skipif(
    not EINSTEINPY_AVAILABLE, reason="EinsteinPy library is not available"
)


# Add a fixture to reset global state between tests
@pytest.fixture(autouse=True)
def reset_globals():
    # Clear global dictionaries before each test
    local_vars.clear()
    expressions.clear()
    if EINSTEINPY_AVAILABLE:
        metrics.clear()
        tensor_objects.clear()
    # Reset the expression counter
    import server

    server.expression_counter = 0
    yield


class TestCreatePredefinedMetric:
    def test_create_schwarzschild_metric(self):
        # Test creating a Schwarzschild metric
        result = create_predefined_metric(Metric.SCHWARZSCHILD)
        assert result == "metric_Schwarzschild"
        assert result in metrics
        assert result in expressions

    def test_create_minkowski_metric(self):
        # Test creating a Minkowski metric
        result = create_predefined_metric(Metric.MINKOWSKI)
        assert result == "metric_Minkowski"
        assert result in metrics
        assert result in expressions

    def test_create_kerr_metric(self):
        # Test creating a Kerr metric
        result = create_predefined_metric(Metric.KERR)
        assert result == "metric_Kerr"
        assert result in metrics
        assert result in expressions

    def test_invalid_metric(self):
        # Try to create a metric that's in the enum but not implemented
        # For this test, we'll assume ALCUBIERRE_WARP is not implemented
        # in the provided metric_map
        result = create_predefined_metric(Metric.ALCUBIERRE_WARP)
        assert "Error" in result
        assert "not implemented" in result


class TestSearchPredefinedMetrics:
    def test_search_with_results(self):
        # Search for metrics containing "Sitter"
        result = search_predefined_metrics("Sitter")
        assert "Found metrics" in result
        assert "DeSitter" in result or "AntiDeSitter" in result

    def test_search_no_results(self):
        # Search for a term unlikely to match any metric
        result = search_predefined_metrics("XYZ123")
        assert "No metrics found" in result


class TestCalculateTensor:
    def test_calculate_ricci_tensor(self):
        # First create a metric
        metric_key = create_predefined_metric(Metric.SCHWARZSCHILD)

        # Calculate Ricci tensor
        result = calculate_tensor(metric_key, Tensor.RICCI_TENSOR)
        assert result == f"riccitensor_{metric_key}"
        assert result in expressions

    def test_calculate_ricci_scalar(self):
        # First create a metric
        metric_key = create_predefined_metric(Metric.SCHWARZSCHILD)

        # Calculate Ricci scalar
        result = calculate_tensor(metric_key, Tensor.RICCI_SCALAR)
        assert result == f"ricciscalar_{metric_key}"
        assert result in expressions

    def test_calculate_einstein_tensor(self):
        # First create a metric
        metric_key = create_predefined_metric(Metric.SCHWARZSCHILD)

        # Calculate Einstein tensor
        result = calculate_tensor(metric_key, Tensor.EINSTEIN_TENSOR)
        assert result == f"einsteintensor_{metric_key}"
        assert result in expressions

    def test_invalid_metric_key(self):
        result = calculate_tensor("nonexistent_metric", Tensor.RICCI_TENSOR)
        assert "Error" in result
        assert "not found" in result

    def test_invalid_tensor_type(self):
        # First create a metric
        metric_key = create_predefined_metric(Metric.SCHWARZSCHILD)

        # Try to calculate a tensor that's in the enum but not implemented
        # This test assumes there's at least one tensor type that's not in the tensor_map
        # If all enums are implemented, this test might need adjustment
        class TestEnum:
            value = "NonExistentTensor"

        result = calculate_tensor(metric_key, TestEnum())
        assert "Error" in result
        assert "not implemented" in result


class TestCreateCustomMetric:
    def test_create_custom_metric(self):
        # Create a simple 2x2 diagonal metric with symbols t, r
        components = [["-1", "0"], ["0", "1"]]
        symbols = ["t", "r"]

        result = create_custom_metric(components, symbols)
        assert result == "metric_custom_0"
        assert result in metrics
        assert result in expressions

    def test_create_custom_minkowski(self):
        # Create a 4x4 Minkowski metric (-1, 1, 1, 1)
        components = [
            ["-1", "0", "0", "0"],
            ["0", "1", "0", "0"],
            ["0", "0", "1", "0"],
            ["0", "0", "0", "1"],
        ]
        symbols = ["t", "x", "y", "z"]

        result = create_custom_metric(components, symbols)
        assert result == "metric_custom_0"
        assert result in metrics
        assert result in expressions

    def test_create_custom_metric_with_expressions(self):
        # Create a metric with symbolic expressions
        components = [
            ["-1", "0", "0", "0"],
            ["0", "r**2", "0", "0"],
            ["0", "0", "r**2 * sin(theta)**2", "0"],
            ["0", "0", "0", "1"],
        ]
        symbols = ["t", "r", "theta", "phi"]

        result = create_custom_metric(components, symbols)
        assert result == "metric_custom_0"
        assert result in metrics
        assert result in expressions

    def test_invalid_components(self):
        # Test with invalid components (not a matrix)
        components = [["1", "0"], ["0"]]  # Missing element in second row
        symbols = ["t", "r"]

        result = create_custom_metric(components, symbols)
        assert "Error" in result


class TestPrintLatexTensor:
    def test_print_metric_latex(self):
        # Create a metric and print it in LaTeX
        metric_key = create_predefined_metric(Metric.MINKOWSKI)

        result = print_latex_tensor(metric_key)
        assert result  # Should return a non-empty string
        assert "\\begin{pmatrix}" in result or "\\left[" in result

    def test_print_tensor_latex(self):
        # Create a metric, calculate a tensor, and print it in LaTeX
        metric_key = create_predefined_metric(Metric.SCHWARZSCHILD)
        tensor_key = calculate_tensor(metric_key, Tensor.RICCI_TENSOR)

        result = print_latex_tensor(tensor_key)
        assert result  # Should return a non-empty string

    def test_nonexistent_tensor(self):
        result = print_latex_tensor("nonexistent_tensor")
        assert "Error" in result
        assert "not found" in result
