"""Metodos de busqueda de raices."""

from __future__ import annotations

from core.utils.metricas import error_absoluto, error_relativo_aproximado
from core.utils.parser_funciones import derive_expression, expression_to_callable
from core.utils.resultados import RootIteration, RootResult
from core.utils.validaciones import validar_intervalo, validar_max_iter, validar_tolerancia


def biseccion(
    f_expr: str,
    a: float,
    b: float,
    tolerancia: float = 1e-6,
    max_iter: int = 100,
    angle_mode: str | None = None,
) -> RootResult:
    """Resuelve f(x)=0 por el metodo de biseccion."""
    validar_tolerancia(tolerancia)
    validar_max_iter(max_iter)
    validar_intervalo(a, b)

    f = expression_to_callable(f_expr, ("x",), angle_mode=angle_mode)
    fa = f(a)
    fb = f(b)

    if fa == 0:
        return RootResult("biseccion", True, a, [], "La raiz coincide con el extremo izquierdo.", {"intervalo": [a, b]})
    if fb == 0:
        return RootResult("biseccion", True, b, [], "La raiz coincide con el extremo derecho.", {"intervalo": [a, b]})
    if fa * fb > 0:
        raise ValueError("No hay cambio de signo en [a, b].")

    iteraciones: list[RootIteration] = []
    izquierda = a
    derecha = b
    f_izquierda = fa

    for iteracion in range(1, max_iter + 1):
        punto_medio = (izquierda + derecha) / 2.0
        f_medio = f(punto_medio)
        err_abs = abs(derecha - izquierda) / 2.0
        error_global = abs(f_medio)
        error_local = err_abs
        iteraciones.append(
            RootIteration(
                iteracion=iteracion,
                aproximacion=punto_medio,
                error_absoluto=err_abs,
                error_relativo=error_relativo_aproximado(punto_medio, iteraciones[-1].aproximacion) if iteraciones else None,
                error_local=error_local,
                error_global=error_global,
                detalles={"a": izquierda, "b": derecha, "f_aprox": f_medio},
            )
        )
        if abs(f_medio) < tolerancia or err_abs < tolerancia:
            return RootResult("biseccion", True, punto_medio, iteraciones, "Metodo convergente.", {"intervalo": [a, b], "tolerancia": tolerancia})
        if f_izquierda * f_medio < 0:
            derecha = punto_medio
        else:
            izquierda = punto_medio
            f_izquierda = f_medio

    return RootResult("biseccion", False, (izquierda + derecha) / 2.0, iteraciones, "Se alcanzo el maximo de iteraciones.", {"intervalo": [a, b], "tolerancia": tolerancia})


def punto_fijo(
    g_expr: str,
    x0: float,
    tolerancia: float = 1e-6,
    max_iter: int = 100,
    angle_mode: str | None = None,
) -> RootResult:
    """Ejecuta la iteracion de punto fijo x_{n+1} = g(x_n)."""
    validar_tolerancia(tolerancia)
    validar_max_iter(max_iter)
    g = expression_to_callable(g_expr, ("x",), angle_mode=angle_mode)

    iteraciones: list[RootIteration] = []
    actual = x0
    for iteracion in range(1, max_iter + 1):
        siguiente = g(actual)
        err_abs = error_absoluto(siguiente, actual)
        error_global = abs(siguiente - actual) # Para punto fijo, el residuo g(x)-x es igual a x_n+1 - x_n
        error_local = err_abs
        iteraciones.append(
            RootIteration(
                iteracion=iteracion,
                aproximacion=siguiente,
                error_absoluto=err_abs,
                error_relativo=error_relativo_aproximado(siguiente, actual),
                error_local=error_local,
                error_global=error_global,
                detalles={"x_anterior": actual},
            )
        )
        if err_abs < tolerancia:
            return RootResult("punto_fijo", True, siguiente, iteraciones, "Metodo convergente.", {"tolerancia": tolerancia})
        actual = siguiente

    return RootResult("punto_fijo", False, actual, iteraciones, "Se alcanzo el maximo de iteraciones.", {"tolerancia": tolerancia})


def newton_raphson(
    f_expr: str,
    x0: float,
    df_expr: str | None = None,
    tolerancia: float = 1e-6,
    max_iter: int = 100,
    angle_mode: str | None = None,
) -> RootResult:
    """Resuelve f(x)=0 por Newton-Raphson."""
    validar_tolerancia(tolerancia)
    validar_max_iter(max_iter)

    derivada_expr = df_expr or derive_expression(f_expr, "x")
    f = expression_to_callable(f_expr, ("x",), angle_mode=angle_mode)
    df = expression_to_callable(derivada_expr, ("x",), angle_mode=angle_mode)

    iteraciones: list[RootIteration] = []
    actual = x0
    for iteracion in range(1, max_iter + 1):
        fx = f(actual)
        dfx = df(actual)
        if abs(dfx) < 1e-14:
            raise ValueError("La derivada es nula o demasiado cercana a cero.")
        siguiente = actual - fx / dfx
        fx_sig = f(siguiente)
        err_abs = error_absoluto(siguiente, actual)
        error_global = abs(fx_sig)
        error_local = err_abs
        iteraciones.append(
            RootIteration(
                iteracion=iteracion,
                aproximacion=siguiente,
                error_absoluto=err_abs,
                error_relativo=error_relativo_aproximado(siguiente, actual),
                error_local=error_local,
                error_global=error_global,
                detalles={"x_anterior": actual, "f_aprox": fx_sig, "f_actual": fx, "df_actual": dfx},
            )
        )
        if err_abs < tolerancia:
            return RootResult("newton_raphson", True, siguiente, iteraciones, "Metodo convergente.", {"tolerancia": tolerancia, "derivada": derivada_expr})
        actual = siguiente

    return RootResult("newton_raphson", False, actual, iteraciones, "Se alcanzo el maximo de iteraciones.", {"tolerancia": tolerancia, "derivada": derivada_expr})
