# Laboratorio de Modelado y Simulación – ModySim Lab

Repositorio Python orientado a métodos numéricos, modelado y simulación con un enfoque doble:

- **Académico**: estudiar, comparar y visualizar métodos numéricos
- **Técnico**: reutilizar el motor para problemas aplicados de ciencia, ingeniería y ciencia de datos

La interfaz web (SPA) se sirve con **Flask** y utiliza un diseño dark-mode premium con teclado matemático virtual, gráficos interactivos (Plotly.js) y panel lateral de teoría con fórmulas renderizadas (KaTeX).

### Estructura principal

- `core/`: motor numérico y utilidades compartidas
- `web/`: interfaz web (HTML/CSS/JS) servida por Flask
- `simuladores/`: interfaz de consola
- `visualizacion/`: gráficos reutilizables
- `casos/`: ejercicios de TP y ejemplos aplicados
- `docs/`: arquitectura, métodos y guías de uso
- `tests/`: pruebas básicas con `pytest`

## Metodos implementados

### Busqueda de raices
- Biseccion
- Punto fijo
- Newton-Raphson
- Aceleracion de Aitken

### Interpolacion y derivacion
- Interpolacion de Lagrange
- Diferencia central

### Integracion numerica
- Trapecio compuesto
- Simpson 1/3 compuesto
- Simpson 3/8 compuesto
- Rectangulo medio compuesto
- Cuadratura de Gauss-Legendre

### Simulacion estocastica
- Monte Carlo para integrales
- Monte Carlo geometrico para estimacion de pi

### EDO
- Euler
- Heun
- Runge-Kutta de orden 4


## Instalación

```bash
# 1. Crear entorno virtual
python -m venv .venv

# 2. Activar entorno virtual
# Windows (PowerShell)
.venv\Scripts\activate
# Windows (CMD)
.venv\Scripts\activate.bat
# Linux / macOS
source .venv/bin/activate

# 3. Instalar dependencias
python -m pip install -r requirements.txt
```

## Ejecución

### Interfaz Web (Flask) — modo recomendado

```bash
# Iniciar el servidor
python web_app.py
```

Abrí el navegador en: **http://127.0.0.1:5000**

Dentro de la web, ahora tenés la opción **"Caso Práctico Integrado"**:

- ejecuta un escenario aplicado de control térmico
- incluye dashboard rediseñado con **Apache ECharts** y KPIs de rendimiento
- permite perfiles de ejecución (Base / Pro / Extremo) para simulaciones más exigentes
- compara métodos de raíces (Bisección, Punto Fijo, Newton, Aitken)
- usa interpolación y derivación numérica sobre datos simulados
- contrasta integración determinística vs Monte Carlo
- resuelve la dinámica con Euler, Heun y RK4

Para **detener** el servidor: presioná `Ctrl + C` en la terminal.

### Consola (modo texto)

```bash
python app.py
```

## Estructura del proyecto

```text
core/
  metodos_numericos/    # Bisección, Newton, Lagrange, Simpson, Monte Carlo, EDO, etc.
  utils/                # Parser de funciones (SymPy)
web/
  templates/            # index.html (SPA)
  static/
    css/style.css       # Diseño dark-mode premium
    js/app.js           # Lógica principal
    js/api.js           # Llamadas REST al backend
    js/math_keyboard.js # Teclado matemático virtual
    js/theory_panel.js  # Panel lateral de teoría
simuladores/
visualizacion/
casos/
docs/
tests/
flask_app.py            # Backend Flask (API REST)
web_app.py              # Entry point
app.py                  # Interfaz de consola
modelos.py              # Fachada de compatibilidad
```

## Stack tecnológico

- Python 3.10+
- Flask (backend + servir SPA)
- Plotly.js (gráficos interactivos)
- KaTeX (renderizado de fórmulas)
- SymPy (parsing y evaluación simbólica)
- Pandas
- Pytest

## Documentación adicional

- [Arquitectura](docs/arquitectura.md)
- [Métodos](docs/metodos.md)
- [Uso](docs/uso.md)
