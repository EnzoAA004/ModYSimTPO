"""Metodos de interpolacion y derivacion numerica."""

from __future__ import annotations

from core.utils.metricas import error_absoluto, error_relativo
from core.utils.parser_funciones import expression_to_callable, interpolate_polynomial_expression
from core.utils.resultados import InterpolationResult, InterpolationStep
from core.utils.validaciones import validar_puntos_interpolacion


def _eval_lagrange_at(x_val: float, puntos: list[tuple[float, float]]) -> float:
    resultado = 0.0
    n = len(puntos)
    for i in range(n):
        xi, yi = puntos[i]
        termino = yi
        for j in range(n):
            if i == j:
                continue
            xj, _ = puntos[j]
            termino *= (x_val - xj) / (xi - xj)
        resultado += termino
    return resultado

def interpolacion_lagrange(
    puntos: list[tuple[float, float]],
    x_eval: float,
    h: float | None = None,
    f_exacta_expr: str | None = None,
    angle_mode: str | None = None,
) -> InterpolationResult:
    """Evalua el polinomio interpolante de Lagrange y opcionalmente tabula."""
    validar_puntos_interpolacion(puntos)

    resultado = _eval_lagrange_at(x_eval, puntos)
    f_exacta = expression_to_callable(f_exacta_expr, ("x",), angle_mode=angle_mode) if f_exacta_expr else None
    
    pasos = []
    if h is not None and h > 0:
        x_min = min(p[0] for p in puntos)
        x_max = max(p[0] for p in puntos)
        actual_x = x_min
        iteracion = 1
        anterior_valor = None
        while actual_x <= x_max + 1e-9:
            val_p = _eval_lagrange_at(actual_x, puntos)
            y_exacto = f_exacta(actual_x) if f_exacta else None
            err_local = error_absoluto(val_p, anterior_valor) if anterior_valor is not None else None
            err_global = error_absoluto(val_p, y_exacto) if y_exacto is not None else None
            err_rel = error_relativo(val_p, y_exacto) if y_exacto is not None else None
            pasos.append(InterpolationStep(iteracion, h, actual_x, val_p, y_exacto, err_local, err_global, err_rel))
            anterior_valor = val_p
            actual_x += h
            iteracion += 1

    return InterpolationResult(
        metodo="lagrange",
        valor_interpolado=resultado,
        x_eval=x_eval,
        puntos=list(puntos),
        mensaje="Interpolacion completada.",
        pasos=pasos,
        metadatos={"polinomio": interpolate_polynomial_expression(puntos)},
    )


def diferencia_central(
    f_expr: str,
    x: float,
    h: float = 1e-4,
    angle_mode: str | None = None,
) -> float:
    """Aproxima la derivada primera mediante diferencia central."""
    if h <= 0:
        raise ValueError("h debe ser mayor a cero.")
    f = expression_to_callable(f_expr, ("x",), angle_mode=angle_mode)
    return (f(x + h) - f(x - h)) / (2.0 * h)
