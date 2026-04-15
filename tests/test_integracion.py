from core.metodos_numericos.integracion import simpson_13_compuesto, trapecio_compuesto


def test_trapecio_compuesto_lineal():
    resultado = trapecio_compuesto("x", 0.0, 1.0, 10, 0.5)
    assert abs(resultado.valor_aproximado - 0.5) < 1e-12
    assert resultado.error_absoluto is not None


def test_simpson_sin():
    resultado = simpson_13_compuesto("sin(x)", 0.0, 3.141592653589793, 10, 2.0)
    assert abs(resultado.valor_aproximado - 2.0) < 1e-3
