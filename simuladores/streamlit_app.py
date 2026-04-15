"""Aplicacion Streamlit del laboratorio numerico."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import streamlit as st

from casos.ejemplos_aplicados import APPLIED_CASES
from casos.ejercicios_tp import INTEGRATION_CASES, INTERPOLATION_CASES, ODE_CASES, ROOT_CASES
from core.metodos_numericos.edo import euler, heun, runge_kutta_4
from core.metodos_numericos.integracion import cuadratura_gauss_legendre, simpson_13_compuesto, simpson_38_compuesto, trapecio_compuesto
from core.metodos_numericos.interpolacion import diferencia_central, interpolacion_lagrange
from core.metodos_numericos.monte_carlo import estimar_pi_geometrico, integracion_montecarlo
from core.metodos_numericos.raices import biseccion, newton_raphson, punto_fijo
from core.metodos_numericos.sistemas_lineales import eliminacion_gauss
from visualizacion.graficos_edo import grafico_trayectoria_edo
from visualizacion.graficos_integracion import grafico_integrando
from visualizacion.graficos_interpolacion import grafico_lagrange
from visualizacion.graficos_montecarlo import histograma_montecarlo
from visualizacion.graficos_raices import grafico_convergencia_raices, grafico_funcion_raiz


def _parsear_puntos(raw: str) -> list[tuple[float, float]]:
    puntos = []
    for bloque in raw.split(";"):
        bloque = bloque.strip()
        if not bloque:
            continue
        x, y = [float(valor.strip()) for valor in bloque.split(",")]
        puntos.append((x, y))
    return puntos


def _mostrar_dataframe(registros: list[dict]) -> None:
    st.dataframe(pd.DataFrame(registros), use_container_width=True)


def _panel_casos() -> None:
    st.subheader("Casos de estudio")
    st.write("Ejemplos listos para usar en trabajos practicos y escenarios aplicados.")
    st.json(
        {
            "raices": list(ROOT_CASES.keys()),
            "interpolacion": list(INTERPOLATION_CASES.keys()),
            "integracion": list(INTEGRATION_CASES.keys()),
            "edo": list(ODE_CASES.keys()),
            "aplicados": list(APPLIED_CASES.keys()),
        }
    )


def _panel_raices() -> None:
    metodo = st.selectbox("Metodo de raices", ["Biseccion", "Punto fijo", "Newton-Raphson"])
    if metodo == "Biseccion":
        f_expr = st.text_input("f(x)", value=ROOT_CASES["biseccion_cos"]["parametros"]["f_expr"])
        a = st.number_input("a", value=0.0)
        b = st.number_input("b", value=1.0)
        tol = st.number_input("Tolerancia", value=1e-6, format="%.10f")
        max_iter = st.number_input("Max iteraciones", value=100, min_value=1)
        if st.button("Ejecutar biseccion", use_container_width=True):
            resultado = biseccion(f_expr, a, b, float(tol), int(max_iter))
            st.success(resultado.mensaje)
            _mostrar_dataframe([asdict(paso) for paso in resultado.iteraciones])
            st.plotly_chart(grafico_convergencia_raices(resultado), use_container_width=True)
            st.plotly_chart(grafico_funcion_raiz(f_expr, resultado.aproximacion, a, b), use_container_width=True)
    elif metodo == "Punto fijo":
        g_expr = st.text_input("g(x)", value=ROOT_CASES["punto_fijo_cos"]["parametros"]["g_expr"])
        x0 = st.number_input("x0", value=0.5)
        tol = st.number_input("Tolerancia PF", value=1e-6, format="%.10f")
        max_iter = st.number_input("Max iter PF", value=100, min_value=1)
        if st.button("Ejecutar punto fijo", use_container_width=True):
            resultado = punto_fijo(g_expr, x0, float(tol), int(max_iter))
            st.success(resultado.mensaje)
            _mostrar_dataframe([asdict(paso) for paso in resultado.iteraciones])
            st.plotly_chart(grafico_convergencia_raices(resultado), use_container_width=True)
    else:
        f_expr = st.text_input("f(x)", value=ROOT_CASES["newton_cubica"]["parametros"]["f_expr"])
        df_expr = st.text_input("f'(x) opcional", value=ROOT_CASES["newton_cubica"]["parametros"]["df_expr"])
        x0 = st.number_input("x0", value=1.5)
        tol = st.number_input("Tolerancia Newton", value=1e-6, format="%.10f")
        max_iter = st.number_input("Max iter Newton", value=100, min_value=1)
        if st.button("Ejecutar Newton", use_container_width=True):
            resultado = newton_raphson(f_expr, x0, df_expr or None, float(tol), int(max_iter))
            st.success(resultado.mensaje)
            _mostrar_dataframe([asdict(paso) for paso in resultado.iteraciones])
            st.plotly_chart(grafico_convergencia_raices(resultado), use_container_width=True)


def _panel_interpolacion() -> None:
    raw = st.text_area("Puntos", value="1,1; 2,4; 3,9")
    x_eval = st.number_input("x a evaluar", value=1.5)
    h = st.number_input("h para derivada central", value=0.01, format="%.10f")
    f_expr = st.text_input("f(x) para derivada opcional", value="sin(x)")
    if st.button("Interpolar", use_container_width=True):
        puntos = _parsear_puntos(raw)
        resultado = interpolacion_lagrange(puntos, x_eval)
        st.success(resultado.mensaje)
        st.write(f"Valor interpolado: `{resultado.valor_interpolado:.10f}`")
        st.write(f"Polinomio: `{resultado.metadatos['polinomio']}`")
        st.plotly_chart(grafico_lagrange(puntos, x_eval), use_container_width=True)
    if st.button("Derivar por diferencia central", use_container_width=True):
        derivada = diferencia_central(f_expr, x_eval, float(h))
        st.info(f"f'({x_eval}) ~= {derivada:.10f}")


def _panel_integracion() -> None:
    metodo = st.selectbox("Metodo de integracion", ["Trapecio compuesto", "Simpson 1/3", "Simpson 3/8", "Gauss-Legendre"])
    f_expr = st.text_input("f(x)", value=INTEGRATION_CASES["simpson_sin"]["parametros"]["f_expr"])
    a = st.number_input("a", value=0.0)
    b = st.number_input("b", value=3.141592653589793)
    n = st.number_input("n / orden", value=10, min_value=1)
    exacto = st.text_input("Valor exacto opcional", value="2.0")
    valor_exacto = float(exacto) if exacto.strip() else None
    if st.button("Integrar", use_container_width=True):
        if metodo == "Trapecio compuesto":
            resultado = trapecio_compuesto(f_expr, a, b, int(n), valor_exacto)
        elif metodo == "Simpson 1/3":
            resultado = simpson_13_compuesto(f_expr, a, b, int(n), valor_exacto)
        elif metodo == "Simpson 3/8":
            resultado = simpson_38_compuesto(f_expr, a, b, int(n), valor_exacto)
        else:
            resultado = cuadratura_gauss_legendre(f_expr, a, b, int(n), valor_exacto)
        st.success(resultado.mensaje)
        st.write(f"Integral aproximada: `{resultado.valor_aproximado:.10f}`")
        if resultado.valor_exacto is not None:
            st.write(f"Error absoluto: `{resultado.error_absoluto:.3e}`")
        _mostrar_dataframe([asdict(muestra) for muestra in resultado.muestras])
        st.plotly_chart(grafico_integrando(f_expr, a, b), use_container_width=True)


def _panel_montecarlo() -> None:
    modo = st.radio("Modo Monte Carlo", ["Integral", "Estimacion de pi"], horizontal=True)
    if modo == "Integral":
        f_expr = st.text_input("f(x)", value="exp(-x**2)")
        a = st.number_input("a", value=0.0)
        b = st.number_input("b", value=1.0)
        n = st.number_input("muestras", value=5000, min_value=10)
        if st.button("Ejecutar Monte Carlo", use_container_width=True):
            resultado = integracion_montecarlo(f_expr, a, b, int(n))
            st.success(resultado.mensaje)
            st.write(f"Estimacion: `{resultado.estimacion:.10f}`")
            st.write(f"IC 95%: `[{resultado.ic_bajo:.10f}, {resultado.ic_alto:.10f}]`")
            st.plotly_chart(histograma_montecarlo(resultado), use_container_width=True)
    else:
        n = st.number_input("muestras para pi", value=10000, min_value=10)
        if st.button("Estimar pi", use_container_width=True):
            resultado = estimar_pi_geometrico(int(n))
            st.success(resultado.mensaje)
            st.write(f"Estimacion de pi: `{resultado.estimacion:.10f}`")
            st.plotly_chart(histograma_montecarlo(resultado), use_container_width=True)


def _panel_edo() -> None:
    metodo = st.selectbox("Metodo EDO", ["Euler", "Heun", "RK4"])
    ode_expr = st.text_input("f(t, y)", value="y")
    t0 = st.number_input("t0", value=0.0)
    y0 = st.number_input("y(t0)", value=1.0)
    tf = st.number_input("tf", value=1.0)
    h = st.number_input("h", value=0.1, format="%.10f")
    if st.button("Resolver EDO", use_container_width=True):
        if metodo == "Euler":
            resultado = euler(ode_expr, t0, y0, float(h), tf=float(tf))
        elif metodo == "Heun":
            resultado = heun(ode_expr, t0, y0, float(h), tf=float(tf))
        else:
            resultado = runge_kutta_4(ode_expr, t0, y0, float(h), tf=float(tf))
        st.success(resultado.mensaje)
        exacta_expr = resultado.metadatos.get("solucion_exacta_expr")
        if exacta_expr:
            st.info(f"Solucion exacta detectada: y(t) = {exacta_expr}")
        _mostrar_dataframe([asdict(paso) for paso in resultado.pasos])
        st.plotly_chart(grafico_trayectoria_edo(resultado), use_container_width=True)


def _panel_sistemas() -> None:
    st.write("Ejemplo base: sistema 2x2.")
    a11 = st.number_input("A11", value=2.0)
    a12 = st.number_input("A12", value=1.0)
    a21 = st.number_input("A21", value=1.0)
    a22 = st.number_input("A22", value=3.0)
    b1 = st.number_input("b1", value=1.0)
    b2 = st.number_input("b2", value=2.0)
    if st.button("Resolver sistema", use_container_width=True):
        resultado = eliminacion_gauss([[a11, a12], [a21, a22]], [b1, b2])
        st.success(resultado.mensaje)
        st.write(f"Solucion: `{resultado.solucion}`")
        st.write(f"Residuo: `{resultado.residuo}`")


def main() -> None:
    st.set_page_config(page_title="Laboratorio de Modelado y Simulacion", layout="wide")
    st.title("Laboratorio de Modelado y Simulacion de Metodos Numericos")
    st.caption("Plataforma academica y tecnica para explorar metodos numericos, simulacion y visualizacion.")

    seccion = st.sidebar.selectbox(
        "Seccion",
        ["Casos", "Raices", "Interpolacion", "Integracion", "Monte Carlo", "EDO", "Sistemas lineales"],
    )
    if seccion == "Casos":
        _panel_casos()
    elif seccion == "Raices":
        _panel_raices()
    elif seccion == "Interpolacion":
        _panel_interpolacion()
    elif seccion == "Integracion":
        _panel_integracion()
    elif seccion == "Monte Carlo":
        _panel_montecarlo()
    elif seccion == "EDO":
        _panel_edo()
    else:
        _panel_sistemas()
