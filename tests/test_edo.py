from core.metodos_numericos.edo import euler, heun, runge_kutta_4


def test_rk4_exp():
    resultado = runge_kutta_4("y", 0.0, 1.0, 0.1, tf=1.0)
    assert abs(resultado.pasos[-1].y - 2.718281828) < 1e-4
    assert resultado.metadatos.get("solucion_exacta_expr") is not None


def test_euler_and_heun_run():
    resultado_euler = euler("y", 0.0, 1.0, 0.1, tf=0.5)
    resultado_heun = heun("y", 0.0, 1.0, 0.1, tf=0.5)
    assert len(resultado_euler.pasos) == 6
    assert len(resultado_heun.pasos) == 6


def test_tf_no_divisible_by_h_reaches_tf():
    resultado = euler("y", 0.0, 1.0, 0.3, tf=1.0)
    assert abs(resultado.pasos[-1].t - 1.0) < 1e-12
    assert resultado.metadatos["pasos"] == 4

