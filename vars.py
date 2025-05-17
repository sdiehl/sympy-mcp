from enum import Enum


class Assumption(Enum):
    ALGEBRAIC = "algebraic"
    COMMUTATIVE = "commutative"
    COMPLEX = "complex"
    EXTENDED_NEGATIVE = "extended_negative"
    EXTENDED_NONNEGATIVE = "extended_nonnegative"
    EXTENDED_NONPOSITIVE = "extended_nonpositive"
    EXTENDED_NONZERO = "extended_nonzero"
    EXTENDED_POSITIVE = "extended_positive"
    EXTENDED_REAL = "extended_real"
    FINITE = "finite"
    HERMITIAN = "hermitian"
    IMAGINARY = "imaginary"
    INFINITE = "infinite"
    INTEGER = "integer"
    IRATIONAL = "irrational"
    NEGATIVE = "negative"
    NONINTEGER = "noninteger"
    NONNEGATIVE = "nonnegative"
    NONPOSITIVE = "nonpositive"
    NONZERO = "nonzero"
    POSITIVE = "positive"
    RATIONAL = "rational"
    REAL = "real"
    TRANSCENDENTAL = "transcendental"
    ZERO = "zero"


class Domain(Enum):
    COMPLEX = "complex"
    REAL = "real"
    INTEGERS = "integers"
    NATURALS = "naturals"


class ODEHint(Enum):
    FACTORABLE = "factorable"
    NTH_ALGEBRAIC = "nth_algebraic"
    SEPARABLE = "separable"
    FIRST_EXACT = "1st_exact"
    FIRST_LINEAR = "1st_linear"
    BERNOULLI = "Bernoulli"
    FIRST_RATIONAL_RICCATI = "1st_rational_riccati"
    RICCATI_SPECIAL_MINUS2 = "Riccati_special_minus2"
    FIRST_HOMOGENEOUS_COEFF_BEST = "1st_homogeneous_coeff_best"
    FIRST_HOMOGENEOUS_COEFF_SUBS_INDEP_DIV_DEP = (
        "1st_homogeneous_coeff_subs_indep_div_dep"
    )
    FIRST_HOMOGENEOUS_COEFF_SUBS_DEP_DIV_INDEP = (
        "1st_homogeneous_coeff_subs_dep_div_indep"
    )
    ALMOST_LINEAR = "almost_linear"
    LINEAR_COEFFICIENTS = "linear_coefficients"
    SEPARABLE_REDUCED = "separable_reduced"
    FIRST_POWER_SERIES = "1st_power_series"
    LIE_GROUP = "lie_group"
    NTH_LINEAR_CONSTANT_COEFF_HOMOGENEOUS = "nth_linear_constant_coeff_homogeneous"
    NTH_LINEAR_EULER_EQ_HOMOGENEOUS = "nth_linear_euler_eq_homogeneous"
    NTH_LINEAR_CONSTANT_COEFF_UNDETERMINED_COEFFICIENTS = (
        "nth_linear_constant_coeff_undetermined_coefficients"
    )
    NTH_LINEAR_EULER_EQ_NONHOMOGENEOUS_UNDETERMINED_COEFFICIENTS = (
        "nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients"
    )
    NTH_LINEAR_CONSTANT_COEFF_VARIATION_OF_PARAMETERS = (
        "nth_linear_constant_coeff_variation_of_parameters"
    )
    NTH_LINEAR_EULER_EQ_NONHOMOGENEOUS_VARIATION_OF_PARAMETERS = (
        "nth_linear_euler_eq_nonhomogeneous_variation_of_parameters"
    )
    LIOUVILLE = "Liouville"
    SECOND_LINEAR_AIRY = "2nd_linear_airy"
    SECOND_LINEAR_BESSEL = "2nd_linear_bessel"
    SECOND_HYPERGEOMETRIC = "2nd_hypergeometric"
    SECOND_HYPERGEOMETRIC_INTEGRAL = "2nd_hypergeometric_Integral"
    NTH_ORDER_REDUCIBLE = "nth_order_reducible"
    SECOND_POWER_SERIES_ORDINARY = "2nd_power_series_ordinary"
    SECOND_POWER_SERIES_REGULAR = "2nd_power_series_regular"
    NTH_ALGEBRAIC_INTEGRAL = "nth_algebraic_Integral"
    SEPARABLE_INTEGRAL = "separable_Integral"
    FIRST_EXACT_INTEGRAL = "1st_exact_Integral"
    FIRST_LINEAR_INTEGRAL = "1st_linear_Integral"
    BERNOULLI_INTEGRAL = "Bernoulli_Integral"
    FIRST_HOMOGENEOUS_COEFF_SUBS_INDEP_DIV_DEP_INTEGRAL = (
        "1st_homogeneous_coeff_subs_indep_div_dep_Integral"
    )
    FIRST_HOMOGENEOUS_COEFF_SUBS_DEP_DIV_INDEP_INTEGRAL = (
        "1st_homogeneous_coeff_subs_dep_div_indep_Integral"
    )
    ALMOST_LINEAR_INTEGRAL = "almost_linear_Integral"
    LINEAR_COEFFICIENTS_INTEGRAL = "linear_coefficients_Integral"
    SEPARABLE_REDUCED_INTEGRAL = "separable_reduced_Integral"
    NTH_LINEAR_CONSTANT_COEFF_VARIATION_OF_PARAMETERS_INTEGRAL = (
        "nth_linear_constant_coeff_variation_of_parameters_Integral"
    )
    NTH_LINEAR_EULER_EQ_NONHOMOGENEOUS_VARIATION_OF_PARAMETERS_INTEGRAL = (
        "nth_linear_euler_eq_nonhomogeneous_variation_of_parameters_Integral"
    )
    LIOUVILLE_INTEGRAL = "Liouville_Integral"
    SECOND_NONLINEAR_AUTONOMOUS_CONSERVED = "2nd_nonlinear_autonomous_conserved"
    SECOND_NONLINEAR_AUTONOMOUS_CONSERVED_INTEGRAL = (
        "2nd_nonlinear_autonomous_conserved_Integral"
    )

class PDEHint(Enum):
    FIRST_LINEAR_CONSTANT_COEFF_HOMOGENEOUS = "1st_linear_constant_coeff_homogeneous"
    FIRST_LINEAR_CONSTANT_COEFF = "1st_linear_constant_coeff"
    FIRST_LINEAR_CONSTANT_COEFF_INTEGRAL = "1st_linear_constant_coeff_Integral"
    FIRST_LINEAR_VARIABLE_COEFF = "1st_linear_variable_coeff"