from server import (
    intro,
    introduce_expression,
    create_matrix,
    matrix_determinant,
    matrix_inverse,
    matrix_eigenvalues,
    matrix_eigenvectors,
    substitute_expression,
    print_latex_expression,
)


def test_matrix_creation():
    # Create a simple 2x2 matrix
    matrix_key = create_matrix([[1, 2], [3, 4]], "M")
    assert matrix_key == "M"


def test_determinant():
    # Create a matrix and calculate its determinant
    matrix_key = create_matrix([[1, 2], [3, 4]], "M")
    det_key = matrix_determinant(matrix_key)
    # Should be -2
    expr = print_latex_expression(det_key)
    assert expr == "-2"


def test_inverse():
    # Create a matrix and calculate its inverse
    matrix_key = create_matrix([[1, 2], [3, 4]], "M")
    inv_key = matrix_inverse(matrix_key)
    # Check result - don't check exact string as it may vary
    expr = print_latex_expression(inv_key)
    # The inverse of [[1, 2], [3, 4]] should have -2, 1, 3/2, -1/2 as elements
    assert "-2" in expr
    assert "1" in expr
    assert "\\frac{3}{2}" in expr


def test_eigenvalues():
    # Create a matrix and calculate its eigenvalues
    matrix_key = create_matrix([[3, 1], [1, 3]], "M")
    evals_key = matrix_eigenvalues(matrix_key)
    # Eigenvalues should be 2 and 4
    expr = print_latex_expression(evals_key)
    assert "2" in expr
    assert "4" in expr


def test_eigenvectors():
    # Create a matrix and calculate its eigenvectors
    matrix_key = create_matrix([[3, 1], [1, 3]], "M")
    evecs_key = matrix_eigenvectors(matrix_key)
    # Just check that the result is not an error
    expr = print_latex_expression(evecs_key)
    assert "Error" not in expr


def test_substitute():
    # Create variables and expressions
    intro("x", [], [])
    intro("y", [], [])
    expr1 = introduce_expression("x**2 + y**2")
    expr2 = introduce_expression("y + 1")
    # Substitute y = y + 1 in x^2 + y^2
    result_key = substitute_expression(expr1, "y", expr2)
    # Result should be x^2 + (y+1)^2 = x^2 + y^2 + 2y + 1
    expr = print_latex_expression(result_key)
    assert "x^{2}" in expr
    assert "y" in expr
