"""Metodos numericos para ecuaciones diferenciales ordinarias."""

from __future__ import annotations

import sympy as sp

from core.utils.errores import ValidationError
from core.utils.metricas import error_absoluto
from core.utils.parser_funciones import expression_to_callable, parse_expression
from core.utils.resultados import ODEResult, ODEStep
from core.utils.validaciones import validar_intervalo, validar_max_iter, validar_paso


def _resolver_edo(
    ode_expr: str,
    t0: float,
    y0: float,
    h: float,
    pasos: int | None,
    tf: float | None,
    solucion_exacta_expr: str | None,
    angle_mode: str | None,
):
    validar_paso(h)
    if tf is None and pasos is None:
        raise ValidationError("Debes indicar tf o pasos para resolver la EDO.")

    hs: list[float] = []
    tf_objetivo: float
    if tf is not None:
        validar_intervalo(t0, tf)
        tf_objetivo = tf
        restante = tf_objetivo - t0
        while restante > 1e-12:
            h_actual = min(h, restante)
            hs.append(h_actual)
            restante -= h_actual
    else:
        assert pasos is not None
        validar_max_iter(pasos)
        hs = [h] * pasos
        tf_objetivo = t0 + pasos * h

    f = expression_to_callable(ode_expr, ("t", "y"), angle_mode=angle_mode)
    exacta_origen = None
    exacta_expr = solucion_exacta_expr
    if solucion_exacta_expr:
        exacta = expression_to_callable(solucion_exacta_expr, ("t",), angle_mode=angle_mode)
        exacta_origen = "manual"
    else:
        exacta_expr, exacta = _resolver_solucion_exacta_automatica(ode_expr, t0, y0)
        if exacta_expr:
            exacta_origen = "automatica"

    metadatos = {
        "h": h,
        "pasos": len(hs),
        "tf": tf_objetivo,
        "h_final": hs[-1] if hs else 0.0,
        "solucion_exacta_expr": exacta_expr,
        "solucion_exacta_origen": exacta_origen,
    }
    return f, exacta, hs, metadatos


def _resolver_solucion_exacta_automatica(ode_expr: str, t0: float, y0: float):
    """Intenta resolver y' = f(t, y) con SymPy y la condición inicial y(t0)=y0."""
    t = sp.Symbol("t")
    y = sp.Function("y")
    try:
        rhs = parse_expression(ode_expr, ("t", "y")).subs({sp.Symbol("y"): y(t)})
        ecuacion = sp.Eq(sp.diff(y(t), t), rhs)
        solucion = sp.dsolve(ecuacion, ics={y(t0): y0})
        rhs_exacto = sp.simplify(solucion.rhs)
        exacta_compilada = sp.lambdify(t, rhs_exacto, modules=["math"])

        def _exacta(t_eval: float) -> float:
            return float(exacta_compilada(t_eval))

        _ = _exacta(t0)
        return str(rhs_exacto), _exacta
    except Exception:  # noqa: BLE001
        return None, None


def euler(
    ode_expr: str,
    t0: float,
    y0: float,
    h: float,
    pasos: int | None = None,
    solucion_exacta_expr: str | None = None,
    angle_mode: str | None = None,
    tf: float | None = None,
) -> ODEResult:
    """Resuelve un PVI por Euler explicito."""
    f, exacta, hs, metadatos = _resolver_edo(ode_expr, t0, y0, h, pasos, tf, solucion_exacta_expr, angle_mode)
    t = t0
    y = y0
    y_exacto = exacta(t) if exacta else None
    error_global = error_absoluto(y, y_exacto) if y_exacto is not None else None
    tabla = [ODEStep(0, t, y, y_exacto, error_global, None, error_global)]
    for paso, h_actual in enumerate(hs, start=1):
        y_prev = y
        y = y + h_actual * f(t, y)
        t = t + h_actual
        y_exacto = exacta(t) if exacta else None
        error_global = error_absoluto(y, y_exacto) if y_exacto is not None else None
        error_local = abs(y - y_prev)
        tabla.append(ODEStep(paso, t, y, y_exacto, error_global, error_local, error_global))
    return ODEResult("euler", tabla, "Integracion completada.", metadatos)


def heun(
    ode_expr: str,
    t0: float,
    y0: float,
    h: float,
    pasos: int | None = None,
    solucion_exacta_expr: str | None = None,
    angle_mode: str | None = None,
    tf: float | None = None,
) -> ODEResult:
    """Resuelve un PVI por el metodo de Heun."""
    f, exacta, hs, metadatos = _resolver_edo(ode_expr, t0, y0, h, pasos, tf, solucion_exacta_expr, angle_mode)
    t = t0
    y = y0
    y_exacto = exacta(t) if exacta else None
    error_global = error_absoluto(y, y_exacto) if y_exacto is not None else None
    tabla = [ODEStep(0, t, y, y_exacto, error_global, None, error_global)]
    for paso, h_actual in enumerate(hs, start=1):
        y_prev = y
        predictor = y + h_actual * f(t, y)
        y = y + h_actual * (f(t, y) + f(t + h_actual, predictor)) / 2.0
        t = t + h_actual
        y_exacto = exacta(t) if exacta else None
        error_global = error_absoluto(y, y_exacto) if y_exacto is not None else None
        error_local = abs(y - y_prev)
        tabla.append(ODEStep(paso, t, y, y_exacto, error_global, error_local, error_global))
    return ODEResult("heun", tabla, "Integracion completada.", metadatos)


def runge_kutta_4(
    ode_expr: str,
    t0: float,
    y0: float,
    h: float,
    pasos: int | None = None,
    solucion_exacta_expr: str | None = None,
    angle_mode: str | None = None,
    tf: float | None = None,
) -> ODEResult:
    """Resuelve un PVI por RK4 clasico."""
    f, exacta, hs, metadatos = _resolver_edo(ode_expr, t0, y0, h, pasos, tf, solucion_exacta_expr, angle_mode)
    t = t0
    y = y0
    y_exacto = exacta(t) if exacta else None
    error_global = error_absoluto(y, y_exacto) if y_exacto is not None else None
    tabla = [ODEStep(0, t, y, y_exacto, error_global, None, error_global)]
    for paso, h_actual in enumerate(hs, start=1):
        y_prev = y
        k1 = f(t, y)
        k2 = f(t + h_actual / 2.0, y + h_actual * k1 / 2.0)
        k3 = f(t + h_actual / 2.0, y + h_actual * k2 / 2.0)
        k4 = f(t + h_actual, y + h_actual * k3)
        y = y + (h_actual / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t = t + h_actual
        y_exacto = exacta(t) if exacta else None
        error_global = error_absoluto(y, y_exacto) if y_exacto is not None else None
        error_local = abs(y - y_prev)
        tabla.append(ODEStep(paso, t, y, y_exacto, error_global, error_local, error_global))
    return ODEResult("runge_kutta_4", tabla, "Integracion completada.", metadatos)
