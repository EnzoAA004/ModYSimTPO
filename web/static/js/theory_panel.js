/**
 * theory_panel.js – Panel lateral de teoría con fórmulas LaTeX.
 *
 * Contiene el contenido teórico de cada sección / método
 * y se actualiza dinámicamente cuando cambia la sección activa.
 */

const TheoryPanel = (() => {
    const panel   = document.getElementById('theory-panel');
    const content = document.getElementById('theory-content');
    const fab     = document.getElementById('theory-fab');
    const toggle  = document.getElementById('theory-toggle');

    let isOpen = false;

    function openPanel() {
        panel.classList.add('open');
        fab.classList.add('hidden');
        isOpen = true;
    }

    function closePanel() {
        panel.classList.remove('open');
        fab.classList.remove('hidden');
        isOpen = false;
    }

    fab.addEventListener('click', openPanel);
    toggle.addEventListener('click', closePanel);

    // --- Contenido teórico por sección ---
    const THEORY = {
        raices: `
            <h4>Búsqueda de Raíces</h4>
            <p>Se busca encontrar <strong>x*</strong> tal que <strong>f(x*) = 0</strong>.</p>

            <h4>Método de Bisección</h4>
            <p>Divide repetidamente un intervalo [a, b] donde hay cambio de signo. En cada iteración el intervalo se reduce a la mitad.</p>
            <div class="formula-block">
                $$c = \\frac{a + b}{2}$$
            </div>
            <p>Si <em>f(a) · f(c) &lt; 0</em>, la raíz está en [a, c]; de lo contrario en [c, b].</p>
            <div class="theory-note">
                <strong>Convergencia:</strong> Siempre converge si hay cambio de signo. Orden lineal, reduce el error a la mitad en cada paso.
            </div>

            <h4>Método de Punto Fijo</h4>
            <p>Se reformula como <strong>x = g(x)</strong> y se itera:</p>
            <div class="formula-block">
                $$x_{n+1} = g(x_n)$$
            </div>
            <div class="theory-note">
                <strong>Condición:</strong> Converge si |g'(x)| &lt; 1 en un entorno de la raíz.
            </div>

            <h4>Newton-Raphson</h4>
            <p>Usa la tangente a la curva para encontrar la raíz con convergencia cuadrática.</p>
            <div class="formula-block">
                $$x_{n+1} = x_n - \\frac{f(x_n)}{f'(x_n)}$$
            </div>
            <div class="theory-note">
                <strong>Requiere:</strong> f'(x) ≠ 0 en la aproximación. Convergencia cuadrática si la raíz es simple.
            </div>

            <h4>Aceleración de Aitken (Δ²)</h4>
            <p>Acelera la convergencia de una sucesión de punto fijo usando tres términos consecutivos:</p>
            <div class="formula-block">
                $$\\hat{x} = x_n - \\frac{(x_{n+1} - x_n)^2}{x_{n+2} - 2x_{n+1} + x_n}$$
            </div>
        `,

        interpolacion: `
            <h4>Interpolación de Lagrange</h4>
            <p>Dado un conjunto de <strong>n+1</strong> puntos, construye el único polinomio de grado ≤ n que pasa por todos ellos.</p>
            <div class="formula-block">
                $$P(x) = \\sum_{i=0}^{n} y_i \\prod_{\\substack{j=0 \\\\ j \\neq i}}^{n} \\frac{x - x_j}{x_i - x_j}$$
            </div>

            <h4>Diferencia Central</h4>
            <p>Aproximación numérica de la primera derivada:</p>
            <div class="formula-block">
                $$f'(x) \\approx \\frac{f(x+h) - f(x-h)}{2h}$$
            </div>
            <div class="theory-note">
                <strong>Error:</strong> Es O(h²), más preciso que diferencias hacia adelante o hacia atrás.
            </div>
        `,

        integracion: `
            <h4>Integración Numérica</h4>
            <p>Aproxima el valor de la integral definida ∫<sub>a</sub><sup>b</sup> f(x) dx.</p>

            <h4>Trapecio Compuesto</h4>
            <div class="formula-block">
                $$\\int_a^b f(x)\\,dx \\approx \\frac{h}{2}\\left[f(a) + 2\\sum_{i=1}^{n-1} f(x_i) + f(b)\\right]$$
            </div>

            <h4>Simpson 1/3 Compuesto</h4>
            <p>Requiere <strong>n par</strong>.</p>
            <div class="formula-block">
                $$\\int_a^b f(x)\\,dx \\approx \\frac{h}{3}\\left[f(a) + 4\\sum_{\\text{impar}} f(x_i) + 2\\sum_{\\text{par}} f(x_i) + f(b)\\right]$$
            </div>

            <h4>Simpson 3/8 Compuesto</h4>
            <p>Requiere <strong>n múltiplo de 3</strong>.</p>
            <div class="formula-block">
                $$\\int_a^b f(x)\\,dx \\approx \\frac{3h}{8}\\left[f(a) + 3\\sum_{i \\not\\equiv 0} f(x_i) + 2\\sum_{i \\equiv 0} f(x_i) + f(b)\\right]$$
            </div>

            <h4>Gauss-Legendre</h4>
            <p>Usa nodos y pesos óptimos (no equiespaciados) para máxima precisión con pocos puntos.</p>
            <div class="formula-block">
                $$\\int_{-1}^{1} f(x)\\,dx \\approx \\sum_{i=1}^{n} w_i \\, f(x_i)$$
            </div>
        `,

        montecarlo: `
            <h4>Método de Monte Carlo</h4>
            <p>Usa muestreo aleatorio para estimar valores numéricos.</p>

            <h4>Integral Monte Carlo</h4>
            <p>Para ∫<sub>a</sub><sup>b</sup> f(x) dx con muestras uniformes en [a, b]:</p>
            <div class="formula-block">
                $$\\hat{I} = \\frac{b-a}{n} \\sum_{i=1}^{n} f(x_i), \\quad x_i \\sim U(a,b)$$
            </div>
            <div class="theory-note">
                <strong>Error:</strong> Decrece como O(1/√n) — independiente de la dimensión.
            </div>

            <h4>Estimación Geométrica de π</h4>
            <p>Se generan puntos aleatorios en el cuadrado [0,1]² y se cuenta cuántos caen dentro del cuarto de círculo:</p>
            <div class="formula-block">
                $$\\pi \\approx 4 \\cdot \\frac{\\text{puntos dentro}}{\\text{total de puntos}}$$
            </div>
        `,

        edo: `
            <h4>Ecuaciones Diferenciales Ordinarias</h4>
            <p>Resuelven un PVI: <strong>y' = f(t, y),   y(t₀) = y₀</strong>.</p>

            <h4>Método de Euler</h4>
            <div class="formula-block">
                $$y_{n+1} = y_n + h \\cdot f(t_n, y_n)$$
            </div>
            <div class="theory-note">Orden 1. Simple pero acumula error rápidamente.</div>

            <h4>Método de Heun (Euler Mejorado)</h4>
            <div class="formula-block">
                $$y_{n+1} = y_n + \\frac{h}{2}\\left[f(t_n, y_n) + f(t_{n+1}, \\tilde{y}_{n+1})\\right]$$
            </div>
            <p>Donde ỹ<sub>n+1</sub> = y<sub>n</sub> + h·f(t<sub>n</sub>, y<sub>n</sub>) es el predictor de Euler.</p>

            <h4>Runge-Kutta de Orden 4 (RK4)</h4>
            <p>El método clásico de orden 4, con excelente balance entre precisión y costo computacional.</p>
            <div class="formula-block">
                $$y_{n+1} = y_n + \\frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$
            </div>
            <ul>
                <li>k₁ = f(tₙ, yₙ)</li>
                <li>k₂ = f(tₙ + h/2, yₙ + h·k₁/2)</li>
                <li>k₃ = f(tₙ + h/2, yₙ + h·k₂/2)</li>
                <li>k₄ = f(tₙ + h, yₙ + h·k₃)</li>
            </ul>
        `,

        'caso-practico': `
            <h4>🔥 Caso práctico: calor de una hornalla (explicado fácil)</h4>
            <p>Este panel muestra algo cotidiano: cómo se reparte el calor alrededor de una hornalla.</p>
            <p>Podés cambiar valores como tamaño de hornalla, potencia o temperatura ambiente y ver al instante qué pasa.</p>

            <div class="theory-step">
                <span class="step-badge">1</span>
                <div>
                    <h4>Distancia segura</h4>
                    <p>El sistema busca a qué distancia la temperatura baja por debajo del valor que marcás como seguro.</p>
                    <div class="formula-block">
                        $$T(r) - T_{segura} = 0$$
                    </div>
                </div>
            </div>

            <div class="theory-step">
                <span class="step-badge">2</span>
                <div>
                    <h4>Curva de temperatura</h4>
                    <p>Con algunos sensores simulados se dibuja una curva continua para entender cómo cae la temperatura al alejarse.</p>
                </div>
            </div>

            <div class="theory-step">
                <span class="step-badge">3</span>
                <div>
                    <h4>Calor total en la zona</h4>
                    <p>Se calcula cuánto calor total hay dentro de un radio dado. Se comparan métodos para ver que dan resultados parecidos.</p>
                </div>
            </div>

            <div class="theory-step">
                <span class="step-badge">4</span>
                <div>
                    <h4>Temperatura en el tiempo</h4>
                    <p>También se simula cómo se calienta una sartén con el paso del tiempo, desde temperatura inicial hasta acercarse a la fuente.</p>
                </div>
            </div>

            <div class="theory-step">
                <span class="step-badge">5</span>
                <div>
                    <h4>Mapa de calor animado</h4>
                    <p>El mapa animado te deja ver de forma visual cómo el calor se expande desde el centro de la hornalla.</p>
                </div>
            </div>

            <div class="theory-note">
                <strong>Idea clave:</strong> si subís la potencia o el radio de la hornalla, el calor llega más lejos.
                Si subís la temperatura segura, la zona segura queda más cerca del centro.
            </div>
        `,
    };

    function update(sectionKey) {
        content.innerHTML = THEORY[sectionKey] || '<p>Selecciona una sección para ver la teoría.</p>';
        // Renderizar KaTeX si está disponible
        if (typeof renderMathInElement === 'function') {
            renderMathInElement(content, {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '$',  right: '$',  display: false },
                ],
                throwOnError: false,
            });
        }
    }

    return { update, openPanel, closePanel };
})();
