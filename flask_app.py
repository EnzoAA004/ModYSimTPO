"""Servidor Flask – API REST para el laboratorio de métodos numéricos."""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import asdict
from typing import Any

import sympy as sp
from flask import Flask, jsonify, render_template, request

from core.metodos_numericos.aitken import aitken_desde_punto_fijo
from core.metodos_numericos.edo import euler, heun, runge_kutta_4
from core.metodos_numericos.integracion import (
    cuadratura_gauss_legendre,
    rectangulo_medio_compuesto,
    simpson_13_compuesto,
    simpson_38_compuesto,
    trapecio_compuesto,
)
from core.metodos_numericos.interpolacion import diferencia_central, interpolacion_lagrange
from core.metodos_numericos.monte_carlo import estimar_pi_geometrico, integracion_montecarlo
from core.metodos_numericos.raices import biseccion, newton_raphson, punto_fijo
# sistemas_lineales removido del alcance del proyecto
from core.utils.parser_funciones import expression_to_callable

_BASE = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(_BASE, "web", "templates"),
    static_folder=os.path.join(_BASE, "web", "static"),
)

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _safe(obj: Any) -> Any:
    """Convierte valores no-JSON (NaN, Inf) a None para serialización segura."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    return obj


def _parse_num(val) -> float:
    """Parsea un valor numérico que puede ser float, int, o string simbólico.

    Soporta expresiones como 'pi', 'e', '2*pi', 'pi/2', 'sqrt(2)', etc.
    """
    if val is None:
        raise ValueError("Se requiere un valor numérico")
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s:
        raise ValueError("Se requiere un valor numérico")
    # Intentar conversión directa primero (rápido)
    try:
        return float(s)
    except ValueError:
        pass
    # Parsear con SymPy para soportar pi, e, sqrt, etc.
    expr = sp.sympify(s, locals={"pi": sp.pi, "e": sp.E, "sqrt": sp.sqrt,
                                  "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                                  "log": sp.log, "exp": sp.exp, "abs": sp.Abs})
    return float(expr.evalf())


def _json_ok(data: dict) -> tuple:
    return jsonify(_safe(data)), 200


def _json_err(msg: str, code: int = 400) -> tuple:
    return jsonify({"error": str(msg)}), code


def _body() -> dict:
    return request.get_json(force=True)


# ---------------------------------------------------------------------------
#  Página principal
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
#  RAÍCES
# ---------------------------------------------------------------------------

@app.route("/api/raices/biseccion", methods=["POST"])
def api_biseccion():
    try:
        d = _body()
        r = biseccion(d["f_expr"], _parse_num(d["a"]), _parse_num(d["b"]),
                       _parse_num(d.get("tolerancia", 1e-6)),
                       int(_parse_num(d.get("max_iter", 100))))
        return _json_ok({
            "convergio": r.convergio,
            "aproximacion": r.aproximacion,
            "mensaje": r.mensaje,
            "iteraciones": [asdict(p) for p in r.iteraciones],
        })
    except Exception as exc:
        return _json_err(exc)


@app.route("/api/raices/punto-fijo", methods=["POST"])
def api_punto_fijo():
    try:
        d = _body()
        r = punto_fijo(d["g_expr"], _parse_num(d["x0"]),
                       _parse_num(d.get("tolerancia", 1e-6)),
                       int(_parse_num(d.get("max_iter", 100))))
        return _json_ok({
            "convergio": r.convergio,
            "aproximacion": r.aproximacion,
            "mensaje": r.mensaje,
            "iteraciones": [asdict(p) for p in r.iteraciones],
        })
    except Exception as exc:
        return _json_err(exc)


@app.route("/api/raices/newton-raphson", methods=["POST"])
def api_newton():
    try:
        d = _body()
        r = newton_raphson(d["f_expr"], _parse_num(d["x0"]),
                           d.get("df_expr") or None,
                           _parse_num(d.get("tolerancia", 1e-6)),
                           int(_parse_num(d.get("max_iter", 100))))
        return _json_ok({
            "convergio": r.convergio,
            "aproximacion": r.aproximacion,
            "mensaje": r.mensaje,
            "iteraciones": [asdict(p) for p in r.iteraciones],
        })
    except Exception as exc:
        return _json_err(exc)


@app.route("/api/raices/aitken", methods=["POST"])
def api_aitken():
    try:
        d = _body()
        r = aitken_desde_punto_fijo(d["g_expr"], _parse_num(d["x0"]),
                                     _parse_num(d.get("tolerancia", 1e-6)),
                                     int(_parse_num(d.get("max_iter", 100))))
        return _json_ok({
            "convergio": r.convergio,
            "aproximacion": r.aproximacion,
            "mensaje": r.mensaje,
            "iteraciones": [asdict(p) for p in r.iteraciones],
        })
    except Exception as exc:
        return _json_err(exc)


# ---------------------------------------------------------------------------
#  INTERPOLACIÓN
# ---------------------------------------------------------------------------

@app.route("/api/interpolacion/lagrange", methods=["POST"])
def api_lagrange():
    try:
        d = _body()
        puntos = [(_parse_num(p[0]), _parse_num(p[1])) for p in d["puntos"]]
        r = interpolacion_lagrange(puntos, _parse_num(d["x_eval"]))
        return _json_ok({
            "valor_interpolado": r.valor_interpolado,
            "mensaje": r.mensaje,
            "polinomio": r.metadatos.get("polinomio", ""),
            "puntos": puntos,
            "x_eval": _parse_num(d["x_eval"]),
        })
    except Exception as exc:
        return _json_err(exc)


@app.route("/api/interpolacion/derivada-central", methods=["POST"])
def api_derivada_central():
    try:
        d = _body()
        valor = diferencia_central(d["f_expr"], _parse_num(d["x"]),
                                    _parse_num(d.get("h", 1e-4)))
        return _json_ok({"derivada": valor, "mensaje": f"f'({d['x']}) ≈ {valor:.10f}"})
    except Exception as exc:
        return _json_err(exc)


# ---------------------------------------------------------------------------
#  INTEGRACIÓN
# ---------------------------------------------------------------------------

def _api_integracion(metodo_fn, d):
    valor_exacto = _parse_num(d["valor_exacto"]) if d.get("valor_exacto") not in (None, "", "null") else None
    r = metodo_fn(d["f_expr"], _parse_num(d["a"]), _parse_num(d["b"]),
                  int(_parse_num(d["n"])), valor_exacto)
    return _json_ok({
        "valor_aproximado": r.valor_aproximado,
        "valor_exacto": r.valor_exacto,
        "error_absoluto": r.error_absoluto,
        "error_relativo": r.error_relativo,
        "mensaje": r.mensaje,
        "muestras": [asdict(m) for m in r.muestras],
    })


@app.route("/api/integracion/trapecio", methods=["POST"])
def api_trapecio():
    try:
        return _api_integracion(trapecio_compuesto, _body())
    except Exception as exc:
        return _json_err(exc)


@app.route("/api/integracion/simpson13", methods=["POST"])
def api_simpson13():
    try:
        return _api_integracion(simpson_13_compuesto, _body())
    except Exception as exc:
        return _json_err(exc)


@app.route("/api/integracion/simpson38", methods=["POST"])
def api_simpson38():
    try:
        return _api_integracion(simpson_38_compuesto, _body())
    except Exception as exc:
        return _json_err(exc)


@app.route("/api/integracion/rectangulo", methods=["POST"])
def api_rectangulo():
    try:
        return _api_integracion(rectangulo_medio_compuesto, _body())
    except Exception as exc:
        return _json_err(exc)


@app.route("/api/integracion/gauss-legendre", methods=["POST"])
def api_gauss():
    try:
        d = _body()
        valor_exacto = _parse_num(d["valor_exacto"]) if d.get("valor_exacto") not in (None, "", "null") else None
        r = cuadratura_gauss_legendre(d["f_expr"], _parse_num(d["a"]), _parse_num(d["b"]),
                                       int(_parse_num(d.get("n", 3))), valor_exacto)
        return _json_ok({
            "valor_aproximado": r.valor_aproximado,
            "valor_exacto": r.valor_exacto,
            "error_absoluto": r.error_absoluto,
            "error_relativo": r.error_relativo,
            "mensaje": r.mensaje,
            "muestras": [asdict(m) for m in r.muestras],
        })
    except Exception as exc:
        return _json_err(exc)


# ---------------------------------------------------------------------------
#  MONTE CARLO
# ---------------------------------------------------------------------------

@app.route("/api/montecarlo/integral", methods=["POST"])
def api_mc_integral():
    try:
        d = _body()
        r = integracion_montecarlo(d["f_expr"], _parse_num(d["a"]), _parse_num(d["b"]),
                                    int(_parse_num(d.get("n", 5000))),
                                    float(d.get("confianza", 0.95)),
                                    int(_parse_num(d["seed"])) if d.get("seed") else None)
        return _json_ok({
            "estimacion": r.estimacion,
            "desvio_muestral": r.desvio_muestral,
            "error_estandar": r.error_estandar,
            "ic_bajo": r.ic_bajo,
            "ic_alto": r.ic_alto,
            "confianza": r.confianza,
            "mensaje": r.mensaje,
            "aportes": [p.aporte for p in r.puntos],
        })
    except Exception as exc:
        return _json_err(exc)


@app.route("/api/montecarlo/pi", methods=["POST"])
def api_mc_pi():
    try:
        d = _body()
        r = estimar_pi_geometrico(int(_parse_num(d.get("n", 10000))),
                                   int(d["seed"]) if d.get("seed") else None)
        return _json_ok({
            "estimacion": r.estimacion,
            "desvio_muestral": r.desvio_muestral,
            "error_estandar": r.error_estandar,
            "ic_bajo": r.ic_bajo,
            "ic_alto": r.ic_alto,
            "confianza": r.confianza,
            "mensaje": r.mensaje,
            "aportes": [p.aporte for p in r.puntos],
        })
    except Exception as exc:
        return _json_err(exc)


# ---------------------------------------------------------------------------
#  EDO
# ---------------------------------------------------------------------------

def _api_edo(metodo_fn, d):
    sol = d.get("solucion_exacta") or None
    r = metodo_fn(d["ode_expr"], _parse_num(d["t0"]), _parse_num(d["y0"]),
                  _parse_num(d["h"]), int(_parse_num(d["pasos"])), sol)
    return _json_ok({
        "mensaje": r.mensaje,
        "pasos": [asdict(p) for p in r.pasos],
    })


@app.route("/api/edo/euler", methods=["POST"])
def api_euler():
    try:
        return _api_edo(euler, _body())
    except Exception as exc:
        return _json_err(exc)


@app.route("/api/edo/heun", methods=["POST"])
def api_heun():
    try:
        return _api_edo(heun, _body())
    except Exception as exc:
        return _json_err(exc)


# ---------------------------------------------------------------------------
#  CASO PRÁCTICO INTEGRADO
# ---------------------------------------------------------------------------

@app.route("/api/casos/practico-integrado", methods=["POST"])
def api_caso_practico_integrado():
    """Ejecuta un caso aplicado que integra todos los algoritmos principales."""
    try:
        d = _body()
        mc_n = int(_parse_num(d.get("mc_n", 6000)))
        pi_n = int(_parse_num(d.get("pi_n", 8000)))
        mc_seed = int(_parse_num(d["seed"])) if d.get("seed") not in (None, "", "null") else 42
        pasos = int(_parse_num(d.get("pasos", 12)))
        h = _parse_num(d.get("h", 0.5))
        cloud_n = int(_parse_num(d.get("cloud_n", 1200)))
        if mc_n < 1000:
            raise ValueError("mc_n debe ser al menos 1000 para el dashboard integrado.")
        if pi_n < 1000:
            raise ValueError("pi_n debe ser al menos 1000 para el dashboard integrado.")
        if cloud_n < 200:
            raise ValueError("cloud_n debe ser al menos 200 para una nube térmica estable.")

        t_ini = time.perf_counter()
        rng = random.Random(mc_seed)

        # Escenario: hornalla domestica con flujo de calor radial.
        ambiente = 24.0
        q0 = 320.0
        alfa = 3.2
        temp_segura = 60.0

        # 1) Raices
        f_balance = "24 + 320*exp(-3.2*x) - 60 - 10*x"
        g_balance = "-(1/3.2)*log((36 + 10*x)/320)"

        r_bis = biseccion(f_balance, 0.0, 2.0, 1e-8, 120)
        r_pf = punto_fijo(g_balance, 0.5, 1e-8, 120)
        r_newton = newton_raphson(f_balance, 0.7, None, 1e-10, 80)
        r_aitken = aitken_desde_punto_fijo(g_balance, 0.5, 1e-10, 80)

        # 2) Interpolacion + derivada radial
        puntos_sensor = [
            (0.00, ambiente + q0 * math.exp(-alfa * 0.00) + 0.0),
            (0.15, ambiente + q0 * math.exp(-alfa * 0.15) + 5.0),
            (0.30, ambiente + q0 * math.exp(-alfa * 0.30) - 3.0),
            (0.45, ambiente + q0 * math.exp(-alfa * 0.45) + 2.0),
            (0.60, ambiente + q0 * math.exp(-alfa * 0.60) - 2.0),
        ]
        temp_r038 = interpolacion_lagrange(puntos_sensor, 0.38)
        gradiente_r03 = diferencia_central("24 + 320*exp(-3.2*x)", 0.30, 1e-4)

        # 3) Integracion (flujo total en disco de radio 0.8 m)
        f_energia = "2*pi*x*320*exp(-3.2*x)"
        a, b = 0.0, 0.8
        i_trap = trapecio_compuesto(f_energia, a, b, 12)
        i_s13 = simpson_13_compuesto(f_energia, a, b, 12)
        i_s38 = simpson_38_compuesto(f_energia, a, b, 12)
        i_rect = rectangulo_medio_compuesto(f_energia, a, b, 12)
        i_gauss = cuadratura_gauss_legendre(f_energia, a, b, 5)

        # 4) Monte Carlo
        mc_i = integracion_montecarlo(f_energia, a, b, mc_n, 0.95, mc_seed)
        mc_pi = estimar_pi_geometrico(pi_n, mc_seed)

        # 5) EDO (calentamiento de una sarten)
        ode_expr = "-0.7*(y-190)"
        t0, y0 = 0.0, 25.0
        exacta = "190 - 165*exp(-0.7*t)"
        edo_e = euler(ode_expr, t0, y0, h, pasos, exacta)
        edo_h = heun(ode_expr, t0, y0, h, pasos, exacta)
        edo_rk4 = runge_kutta_4(ode_expr, t0, y0, h, pasos, exacta)

        # 6) Visualizaciones (nube de puntos + perfil radial + mapa termico)
        nube_puntos = []
        for _ in range(cloud_n):
            x = rng.uniform(-0.85, 0.85)
            y = rng.uniform(-0.85, 0.85)
            r = math.sqrt(x * x + y * y)
            t = ambiente + q0 * math.exp(-alfa * r) + rng.uniform(-4.0, 4.0)
            nube_puntos.append([x, y, t])

        rs = [0.8 * i / 120.0 for i in range(121)]
        ts = [ambiente + q0 * math.exp(-alfa * r) for r in rs]

        # Grilla continua para heatmap 2D
        grid_n = 49
        xs = [-0.9 + (1.8 * i / (grid_n - 1)) for i in range(grid_n)]
        ys = [-0.9 + (1.8 * j / (grid_n - 1)) for j in range(grid_n)]

        # Mapa estacionario
        heatmap_estatico = []
        for x in xs:
            for y in ys:
                r = math.sqrt(x * x + y * y)
                t = ambiente + q0 * math.exp(-alfa * r)
                heatmap_estatico.append([x, y, t])

        # Animacion temporal: encendido progresivo de la hornalla
        beta = 0.9
        tiempos_anim = [10.0 * k / 15.0 for k in range(16)]
        frames_anim = []
        for tt in tiempos_anim:
            escala = 1.0 - math.exp(-beta * tt)
            frame = []
            for x in xs:
                for y in ys:
                    r = math.sqrt(x * x + y * y)
                    t = ambiente + (q0 * escala) * math.exp(-alfa * r)
                    frame.append([x, y, t])
            frames_anim.append({"t": tt, "data": frame})

        elapsed_ms = (time.perf_counter() - t_ini) * 1000.0

        def _ultimo_error(edo_result):
            ultimo = edo_result.pasos[-1]
            return ultimo.error_absoluto

        return _json_ok(
            {
                "caso": {
                    "titulo": "Hornalla doméstica: flujo de calor radial",
                    "descripcion": "Caso práctico con sensores, zona segura, potencia térmica total y calentamiento temporal.",
                    "runtime_ms": elapsed_ms,
                },
                "aplicacion": {
                    "ambiente_c": ambiente,
                    "temperatura_segura_c": temp_segura,
                    "distancia_segura_m": r_newton.aproximacion,
                },
                "raices": {
                    "biseccion": {
                        "aproximacion": r_bis.aproximacion,
                        "iteraciones": len(r_bis.iteraciones),
                        "convergio": r_bis.convergio,
                    },
                    "punto_fijo": {
                        "aproximacion": r_pf.aproximacion,
                        "iteraciones": len(r_pf.iteraciones),
                        "convergio": r_pf.convergio,
                    },
                    "newton_raphson": {
                        "aproximacion": r_newton.aproximacion,
                        "iteraciones": len(r_newton.iteraciones),
                        "convergio": r_newton.convergio,
                    },
                    "aitken": {
                        "aproximacion": r_aitken.aproximacion,
                        "iteraciones": len(r_aitken.iteraciones),
                        "convergio": r_aitken.convergio,
                    },
                    "serie_errores": {
                        "biseccion": [it.error_absoluto for it in r_bis.iteraciones if it.error_absoluto is not None],
                        "punto_fijo": [it.error_absoluto for it in r_pf.iteraciones if it.error_absoluto is not None],
                        "newton": [it.error_absoluto for it in r_newton.iteraciones if it.error_absoluto is not None],
                        "aitken": [it.error_absoluto for it in r_aitken.iteraciones if it.error_absoluto is not None],
                    },
                },
                "interpolacion": {
                    "puntos": puntos_sensor,
                    "x_eval": temp_r038.x_eval,
                    "valor_interpolado": temp_r038.valor_interpolado,
                    "polinomio": temp_r038.metadatos.get("polinomio", ""),
                    "derivada_t3": gradiente_r03,
                },
                "integracion": {
                    "intervalo": [a, b],
                    "resultados": {
                        "trapecio": i_trap.valor_aproximado,
                        "simpson13": i_s13.valor_aproximado,
                        "simpson38": i_s38.valor_aproximado,
                        "rectangulo": i_rect.valor_aproximado,
                        "gauss_legendre": i_gauss.valor_aproximado,
                    },
                },
                "montecarlo": {
                    "integral": {
                        "estimacion": mc_i.estimacion,
                        "ic_bajo": mc_i.ic_bajo,
                        "ic_alto": mc_i.ic_alto,
                        "confianza": mc_i.confianza,
                        "n": mc_n,
                    },
                    "pi": {
                        "estimacion": mc_pi.estimacion,
                        "ic_bajo": mc_pi.ic_bajo,
                        "ic_alto": mc_pi.ic_alto,
                        "confianza": mc_pi.confianza,
                        "n": pi_n,
                    },
                },
                "edo": {
                    "metodos": {
                        "euler": {
                            "y_final": edo_e.pasos[-1].y,
                            "error_final": _ultimo_error(edo_e),
                        },
                        "heun": {
                            "y_final": edo_h.pasos[-1].y,
                            "error_final": _ultimo_error(edo_h),
                        },
                        "rk4": {
                            "y_final": edo_rk4.pasos[-1].y,
                            "error_final": _ultimo_error(edo_rk4),
                        },
                    },
                    "trayectorias": {
                        "t": [p.t for p in edo_rk4.pasos],
                        "euler": [p.y for p in edo_e.pasos],
                        "heun": [p.y for p in edo_h.pasos],
                        "rk4": [p.y for p in edo_rk4.pasos],
                        "exacta": [p.y_exacto for p in edo_rk4.pasos],
                    },
                },
                "visualizacion": {
                    "nube_puntos": nube_puntos,
                    "curva_radial": {"r": rs, "temp": ts},
                    "heatmap_estatico": heatmap_estatico,
                    "heatmap_animado": {
                        "tiempos": tiempos_anim,
                        "frames": frames_anim,
                        "grid_n": grid_n,
                    },
                },
                "mensaje": "Caso práctico integrado ejecutado correctamente.",
            }
        )
    except Exception as exc:
        return _json_err(exc)


@app.route("/api/edo/rk4", methods=["POST"])
def api_rk4():
    try:
        return _api_edo(runge_kutta_4, _body())
    except Exception as exc:
        return _json_err(exc)





# ---------------------------------------------------------------------------
#  UTILIDAD – evaluar curva para gráficos frontend
# ---------------------------------------------------------------------------

@app.route("/api/util/evaluar-curva", methods=["POST"])
def api_evaluar_curva():
    """Evalúa f(x) en un rango para dibujar la curva en Plotly.js."""
    try:
        d = _body()
        f = expression_to_callable(d["f_expr"], ("x",))
        a = _parse_num(d["a"])
        b = _parse_num(d["b"])
        n_puntos = int(d.get("n_puntos", 300))
        xs = [a + (b - a) * i / n_puntos for i in range(n_puntos + 1)]
        ys = []
        for x in xs:
            try:
                ys.append(f(x))
            except Exception:
                ys.append(None)
        return _json_ok({"xs": xs, "ys": ys})
    except Exception as exc:
        return _json_err(exc)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)
