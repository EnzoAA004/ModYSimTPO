# Uso del laboratorio

## Expresiones matematicas

Se aceptan expresiones como:

- `exp(x) - 2 - x`
- `cos(x) - x`
- `x**2 - 4`
- `sin(x)`

Funciones soportadas:

- `sin`, `cos`, `tan`
- `exp`, `log`, `sqrt`
- `abs`
- constantes `pi` y `e`

## Uso desde Python

```python
from core.metodos_numericos.raices import biseccion

resultado = biseccion("cos(x) - x", 0.0, 1.0, 1e-6, 100)
print(resultado.aproximacion)
```

## Uso desde notebooks

```python
from simuladores.notebook_helpers import runge_kutta_4

resultado = runge_kutta_4("y", 0.0, 1.0, 0.1, 10, "exp(t)")
```

## Uso desde consola

```bash
python app.py
```

## Uso desde Streamlit

```bash
streamlit run simuladores/streamlit_app.py
```

## Uso del Caso Práctico Integrado (Web)

1. Levantar la web con Flask.
2. Ir a la sección "Caso Práctico Integrado" en el menú lateral.
3. Ejecutar el botón "Ejecutar Caso Práctico Integrado".

La corrida devuelve un tablero con:

- convergencia de métodos de raíces
- comparación de métodos de integración
- estimaciones Monte Carlo (integral y pi)
- trayectoria EDO para Euler, Heun y RK4
