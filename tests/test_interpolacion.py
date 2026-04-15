from core.metodos_numericos.interpolacion import diferencia_central, interpolacion_lagrange


def test_lagrange_cuadratico():
    resultado = interpolacion_lagrange([(1.0, 1.0), (2.0, 4.0), (3.0, 9.0)], 1.5, h=1.0, f_exacta_expr="x**2")
    assert abs(resultado.valor_interpolado - 2.25) < 1e-10
    assert len(resultado.pasos) == 3
    assert resultado.pasos[0].h == 1.0
    assert resultado.pasos[0].y_exacto == 1.0
    assert resultado.pasos[0].error_global == 0.0
    assert resultado.pasos[-1].error_global == 0.0


def test_diferencia_central_sin():
    derivada = diferencia_central("sin(x)", 0.0, 1e-5)
    assert abs(derivada - 1.0) < 1e-5
