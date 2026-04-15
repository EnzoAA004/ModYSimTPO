/**
 * app.js – Lógica principal de la SPA v2.
 *
 * Mejoras:
 *  – Gráficos Plotly con diseño premium (gradientes, sombras, mejor tipografía)
 *  – Mejor renderizado de tablas
 *  – Loading skeleton animado
 *  – UX más pulido
 */

(() => {
    'use strict';

    // ==================== Helpers ====================

    const $ = (sel, ctx = document) => ctx.querySelector(sel);
    const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];
    const on = (sel, event, handler, ctx = document) => {
        const el = $(sel, ctx);
        if (el) el.addEventListener(event, handler);
        return el;
    };

    /** Plotly dark layout – premium version */
    const PLOTLY_LAYOUT = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor:  'rgba(12,12,20,0.5)',
        font: { family: 'Inter, sans-serif', color: '#8892a4', size: 12 },
        margin: { t: 50, r: 25, b: 50, l: 60 },
        xaxis: {
            gridcolor: 'rgba(255,255,255,0.035)',
            zerolinecolor: 'rgba(255,255,255,0.08)',
            linecolor: 'rgba(255,255,255,0.06)',
            tickfont: { size: 11, family: 'JetBrains Mono, monospace' },
        },
        yaxis: {
            gridcolor: 'rgba(255,255,255,0.035)',
            zerolinecolor: 'rgba(255,255,255,0.08)',
            linecolor: 'rgba(255,255,255,0.06)',
            tickfont: { size: 11, family: 'JetBrains Mono, monospace' },
        },
        colorway: ['#06b6d4', '#8b5cf6', '#22c55e', '#f59e0b', '#ef4444', '#ec4899'],
        hoverlabel: {
            bgcolor: 'rgba(14,14,24,0.92)',
            bordercolor: 'rgba(6,182,212,0.3)',
            font: { family: 'Inter, sans-serif', size: 13, color: '#f0f0f5' },
        },
        legend: {
            bgcolor: 'rgba(0,0,0,0)',
            bordercolor: 'rgba(255,255,255,0.06)',
            borderwidth: 1,
            font: { size: 12, color: '#9ca3af' },
            x: 1, xanchor: 'right', y: 1.05,
            orientation: 'h',
        },
    };
    const PLOTLY_CFG = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
        displaylogo: false,
        modeBarStyle: { bgcolor: 'transparent', color: '#6b7280', activecolor: '#06b6d4' },
    };

    function plotlyLayout(extra = {}) {
        const base = JSON.parse(JSON.stringify(PLOTLY_LAYOUT));
        return mergeDeep(base, extra);
    }

    function mergeDeep(target, source) {
        for (const key of Object.keys(source)) {
            if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
                target[key] = target[key] || {};
                mergeDeep(target[key], source[key]);
            } else {
                target[key] = source[key];
            }
        }
        return target;
    }

    /** Toast notification */
    function toast(msg, type = 'info') {
        const container = $('#toast-container');
        const el = document.createElement('div');
        el.className = `toast ${type}`;
        el.textContent = msg;
        container.appendChild(el);
        setTimeout(() => { if (el.parentNode) el.remove(); }, 3800);
    }

    /** Render HTML table from array of objects */
    function renderTable(rows, columns = null) {
        if (!rows || rows.length === 0) return '<p style="color:var(--text-muted);padding:1rem;">Sin datos.</p>';
        const keys = columns || Object.keys(rows[0]);

        const prettyName = (k) => {
            const map = {
                iteracion: 'Iteración', aproximacion: 'Aproximación',
                error_absoluto: 'Error Abs', error_relativo: 'Error Rel',
                indice: '#', x: 'x', fx: 'f(x)', peso: 'Peso', aporte: 'Aporte',
                paso: 'Paso', t: 't', y: 'y', y_exacto: 'y exacto',
            };
            return map[k] || k;
        };

        const formatNum = (v) => {
            if (v === null || v === undefined) return '<span style="color:var(--text-muted)">—</span>';
            if (typeof v === 'number') {
                if (Number.isInteger(v)) return v.toString();
                if (Math.abs(v) < 0.001 || Math.abs(v) > 99999) return v.toExponential(6);
                return v.toFixed(10);
            }
            if (typeof v === 'object') return JSON.stringify(v);
            return String(v);
        };

        const header = keys.map(k => `<th>${prettyName(k)}</th>`).join('');
        const body = rows.map((r, idx) =>
            `<tr>${keys.map(k => `<td>${formatNum(r[k])}</td>`).join('')}</tr>`
        ).join('');
        return `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
    }

    /** Set loading state on button */
    function setLoading(btn, loading) {
        if (loading) {
            btn.classList.add('loading');
            btn._origText = btn.innerHTML;
            btn.innerHTML = '<span class="btn-icon">⏳</span> Calculando…';
        } else {
            btn.classList.remove('loading');
            if (btn._origText) btn.innerHTML = btn._origText;
        }
    }

    /** Show result message */
    function showMessage(el, msg, isSuccess = true, extra = '') {
        el.className = `result-message ${isSuccess ? 'success' : 'error'}`;
        el.innerHTML = msg + (extra ? `<span class="approx-value">${extra}</span>` : '');
    }

    // ==================== Navigation ====================

    const SECTIONS = {
        raices:        { title: 'Búsqueda de Raíces' },
        interpolacion: { title: 'Interpolación y Derivación' },
        integracion:   { title: 'Integración Numérica' },
        montecarlo:    { title: 'Simulación Monte Carlo' },
        edo:           { title: 'Ecuaciones Diferenciales' },
        'caso-practico': { title: 'Caso Práctico Integrado' },
    };

    function switchSection(key) {
        $$('.nav-btn').forEach(b => b.classList.toggle('active', b.dataset.section === key));
        $$('.section').forEach(s => s.classList.toggle('active', s.id === `sec-${key}`));
        $('#section-title').textContent = SECTIONS[key]?.title || key;
        TheoryPanel.update(key);
        $('#sidebar').classList.remove('open');
    }

    $$('.nav-btn').forEach(btn => {
        btn.addEventListener('click', () => switchSection(btn.dataset.section));
    });

    on('#menu-toggle', 'click', () => {
        $('#sidebar').classList.toggle('open');
    });

    // ==================== RAÍCES ====================

    const raicesMetodo = $('#raices-metodo');
    if (raicesMetodo) {
        raicesMetodo.addEventListener('change', () => {
            const m = raicesMetodo.value;
            const secRaices = $('#sec-raices');
            if (!secRaices) return;
            $$('.method-fields', secRaices).forEach(f => f.style.display = 'none');
            const target = $(`#raices-fields-${m}`);
            if (target) target.style.display = '';
        });
    }

    on('#btn-raices', 'click', async () => {
        const btn = $('#btn-raices');
        if (!btn || !raicesMetodo) return;
        const metodo = raicesMetodo.value;
        let payload;

        try {
            const tol = $('#raices-tol').value;
            const maxIter = $('#raices-maxiter').value;

            if (metodo === 'biseccion') {
                payload = {
                    f_expr: $('#raices-f-expr').value,
                    a: $('#raices-a').value,
                    b: $('#raices-b').value,
                    tolerancia: tol, max_iter: maxIter,
                };
            } else if (metodo === 'punto-fijo') {
                payload = {
                    g_expr: $('#raices-g-expr').value,
                    x0: $('#raices-pf-x0').value,
                    tolerancia: tol, max_iter: maxIter,
                };
            } else if (metodo === 'newton-raphson') {
                payload = {
                    f_expr: $('#raices-nr-f').value,
                    df_expr: $('#raices-nr-df').value || '',
                    x0: $('#raices-nr-x0').value,
                    tolerancia: tol, max_iter: maxIter,
                };
            } else {
                payload = {
                    g_expr: $('#raices-aitken-g').value,
                    x0: $('#raices-aitken-x0').value,
                    tolerancia: tol, max_iter: maxIter,
                };
            }

            setLoading(btn, true);
            $('#status-text').textContent = 'Calculando…';

            const apiFn = {
                'biseccion': API.biseccion,
                'punto-fijo': API.puntoFijo,
                'newton-raphson': API.newtonRaphson,
                'aitken': API.aitken,
            }[metodo];

            const res = await apiFn(payload);
            const area = $('#raices-resultado');
            area.style.display = '';

            showMessage($('#raices-msg'), res.mensaje, res.convergio, `x* ≈ ${res.aproximacion}`);

            // Table
            const cols = ['iteracion', 'aproximacion', 'error_absoluto'];
            $('#raices-tabla').innerHTML = renderTable(res.iteraciones, cols);

            // Convergence chart (log scale for better visualization)
            const iters = res.iteraciones.map(p => p.iteracion);
            const errors = res.iteraciones.map(p => p.error_absoluto);
            Plotly.newPlot('raices-chart', [{
                x: iters, y: errors,
                mode: 'lines+markers',
                name: 'Error absoluto',
                line: { color: '#06b6d4', width: 2.5, shape: 'spline' },
                marker: { color: '#8b5cf6', size: 6, line: { color: '#06b6d4', width: 1 } },
                fill: 'tozeroy',
                fillcolor: 'rgba(6,182,212,0.06)',
            }], plotlyLayout({
                title: { text: 'Convergencia del Método', font: { size: 15, color: '#f0f0f5' } },
                xaxis: { title: { text: 'Iteración', font: { size: 12 } } },
                yaxis: { title: { text: 'Error absoluto', font: { size: 12 } }, type: errors[0] > 1 ? 'linear' : 'log' },
            }), PLOTLY_CFG);

            toast('Cálculo completado ✓', 'success');
        } catch (err) {
            toast(err.message, 'error');
            const area = $('#raices-resultado');
            area.style.display = '';
            showMessage($('#raices-msg'), err.message, false);
        } finally {
            setLoading(btn, false);
            $('#status-text').textContent = 'Listo';
        }
    });

    // ==================== INTERPOLACIÓN – Editor dinámico de puntos ====================

    const pointsList = $('#interp-points-list');
    const defaultPoints = [[1, 1], [2, 4], [3, 9]];

    function addPointRow(x = '', y = '') {
        if (!pointsList) return;
        const idx = pointsList.children.length + 1;
        const row = document.createElement('div');
        row.className = 'point-row';
        row.innerHTML = `
            <span class="point-label">P${idx}</span>
            <div class="point-inputs">
                <input type="text" class="interp-x" data-keyboard="true" value="${x}" placeholder="x">
                <input type="text" class="interp-y" data-keyboard="true" value="${y}" placeholder="y">
            </div>
            <button type="button" class="btn-remove-point" title="Quitar punto">✕</button>
        `;
        row.querySelector('.btn-remove-point').addEventListener('click', () => {
            if (pointsList.children.length > 2) {
                row.remove();
                renumberPoints();
            }
        });
        pointsList.appendChild(row);
        // Re-inyectar botones ⌨ para los nuevos inputs
        if (typeof MathKeyboard !== 'undefined') MathKeyboard.injectToggleButtons();
    }

    function renumberPoints() {
        if (!pointsList) return;
        Array.from(pointsList.children).forEach((row, i) => {
            row.querySelector('.point-label').textContent = `P${i + 1}`;
        });
    }

    function getPointsFromEditor() {
        if (!pointsList) return [];
        return Array.from(pointsList.querySelectorAll('.point-row')).map(row => {
            const x = parseFloat(row.querySelector('.interp-x').value);
            const y = parseFloat(row.querySelector('.interp-y').value);
            return [x, y];
        }).filter(p => !isNaN(p[0]) && !isNaN(p[1]));
    }

    // Init default points
    if (pointsList) {
        defaultPoints.forEach(([x, y]) => addPointRow(x, y));
    }

    on('#interp-add-point', 'click', () => addPointRow());

    on('#btn-interp', 'click', async () => {
        const btn = $('#btn-interp');
        if (!btn) return;
        try {
            const puntos = getPointsFromEditor();
            if (puntos.length < 2) {
                toast('Se necesitan al menos 2 puntos válidos', 'error');
                return;
            }

            setLoading(btn, true);
            $('#status-text').textContent = 'Calculando…';

            const xEval = $('#interp-xeval').value;
            const res = await API.lagrange({ puntos, x_eval: xEval });
            const area = $('#interp-resultado');
            area.style.display = '';

            showMessage($('#interp-msg'), res.mensaje, true,
                `P(${res.x_eval}) = ${res.valor_interpolado.toFixed(10)}`);

            // Chart: polynomial curve + nodes + evaluation point
            const xs_data = puntos.map(p => p[0]);
            const xmin = Math.min(...xs_data) - 0.8;
            const xmax = Math.max(...xs_data) + 0.8;

            let curva;
            try {
                curva = await API.evaluarCurva({ f_expr: res.polinomio, a: xmin, b: xmax, n_puntos: 250 });
            } catch { curva = null; }

            const traces = [];
            if (curva) {
                traces.push({
                    x: curva.xs, y: curva.ys,
                    mode: 'lines', name: 'Polinomio interpolante',
                    line: { color: '#06b6d4', width: 2.5, shape: 'spline' },
                    fill: 'tozeroy',
                    fillcolor: 'rgba(6,182,212,0.04)',
                });
            }
            traces.push({
                x: xs_data, y: puntos.map(p => p[1]),
                mode: 'markers', name: 'Nodos',
                marker: { color: '#8b5cf6', size: 10, line: { color: '#fff', width: 1.5 } },
            });
            traces.push({
                x: [xEval], y: [res.valor_interpolado],
                mode: 'markers', name: `P(${xEval})`,
                marker: { color: '#22c55e', size: 14, symbol: 'diamond', line: { color: '#fff', width: 2 } },
            });

            Plotly.newPlot('interp-chart', traces, plotlyLayout({
                title: { text: 'Interpolación de Lagrange', font: { size: 15, color: '#f0f0f5' } },
                xaxis: { title: { text: 'x' } },
                yaxis: { title: { text: 'P(x)' } },
            }), PLOTLY_CFG);

            toast('Interpolación completada ✓', 'success');
        } catch (err) {
            toast(err.message, 'error');
            const area = $('#interp-resultado');
            area.style.display = '';
            showMessage($('#interp-msg'), err.message, false);
        } finally {
            setLoading(btn, false);
            $('#status-text').textContent = 'Listo';
        }
    });

    on('#btn-deriv', 'click', async () => {
        const btn = $('#btn-deriv');
        if (!btn) return;
        try {
            setLoading(btn, true);
            const res = await API.derivadaCentral({
                f_expr: $('#deriv-expr').value,
                x: $('#deriv-x').value,
                h: $('#deriv-h').value,
            });
            const area = $('#interp-resultado');
            area.style.display = '';
            showMessage($('#interp-msg'), res.mensaje, true);
            toast('Derivada calculada ✓', 'success');
        } catch (err) {
            toast(err.message, 'error');
        } finally {
            setLoading(btn, false);
            $('#status-text').textContent = 'Listo';
        }
    });

    // ==================== INTEGRACIÓN ====================

    on('#btn-integ', 'click', async () => {
        const btn = $('#btn-integ');
        if (!btn) return;
        const metodo = $('#integ-metodo').value;

        try {
            const payload = {
                f_expr: $('#integ-f').value,
                a: $('#integ-a').value,
                b: $('#integ-b').value,
                n: $('#integ-n').value,
                valor_exacto: $('#integ-exacto').value.trim() || null,
            };

            setLoading(btn, true);
            $('#status-text').textContent = 'Calculando…';

            const apiFn = {
                'trapecio': API.trapecio, 'simpson13': API.simpson13,
                'simpson38': API.simpson38, 'rectangulo': API.rectangulo,
                'gauss-legendre': API.gaussLegendre,
            }[metodo];

            const res = await apiFn(payload);
            const area = $('#integ-resultado');
            area.style.display = '';

            let extra = `∫ f(x)dx ≈ ${res.valor_aproximado.toFixed(10)}`;
            if (res.error_absoluto != null) extra += `  |  Error: ${res.error_absoluto.toExponential(4)}`;
            showMessage($('#integ-msg'), res.mensaje, true, extra);

            // Table
            $('#integ-tabla').innerHTML = renderTable(res.muestras, ['indice', 'x', 'fx', 'peso', 'aporte']);

            // Graph: f(x) with filled area
            const curva = await API.evaluarCurva({ f_expr: payload.f_expr, a: payload.a, b: payload.b, n_puntos: 300 });
            Plotly.newPlot('integ-chart', [{
                x: curva.xs, y: curva.ys,
                mode: 'lines',
                fill: 'tozeroy',
                fillcolor: 'rgba(139, 92, 246, 0.12)',
                line: { color: '#8b5cf6', width: 2.5, shape: 'spline' },
                name: 'f(x)',
            }, {
                x: res.muestras.map(m => m.x), y: res.muestras.map(m => m.fx),
                mode: 'markers', name: 'Nodos',
                marker: { color: '#06b6d4', size: 7, line: { color: '#fff', width: 1 } },
            }], plotlyLayout({
                title: { text: `Integrando – ${metodo}`, font: { size: 15, color: '#f0f0f5' } },
                xaxis: { title: { text: 'x' } },
                yaxis: { title: { text: 'f(x)' } },
                shapes: [{
                    type: 'line', x0: payload.a, x1: payload.a, y0: 0, y1: 1, yref: 'paper',
                    line: { color: 'rgba(245,158,11,0.4)', width: 1, dash: 'dash' },
                }, {
                    type: 'line', x0: payload.b, x1: payload.b, y0: 0, y1: 1, yref: 'paper',
                    line: { color: 'rgba(245,158,11,0.4)', width: 1, dash: 'dash' },
                }],
            }), PLOTLY_CFG);

            toast('Integración completada ✓', 'success');
        } catch (err) {
            toast(err.message, 'error');
            const area = $('#integ-resultado');
            area.style.display = '';
            showMessage($('#integ-msg'), err.message, false);
        } finally {
            setLoading(btn, false);
            $('#status-text').textContent = 'Listo';
        }
    });

    // ==================== MONTE CARLO ====================

    const mcModo = $('#mc-modo');
    if (mcModo) {
        mcModo.addEventListener('change', () => {
            const mcFields = $('#mc-fields-integral');
            if (mcFields) mcFields.style.display = mcModo.value === 'integral' ? '' : 'none';
        });
    }

    on('#btn-mc', 'click', async () => {
        const btn = $('#btn-mc');
        if (!btn || !mcModo) return;
        const modo = mcModo.value;

        try {
            setLoading(btn, true);
            $('#status-text').textContent = 'Simulando…';

            let res;
            if (modo === 'integral') {
                res = await API.mcIntegral({
                    f_expr: $('#mc-f').value,
                    a: $('#mc-a').value,
                    b: $('#mc-b').value,
                    n: $('#mc-n').value,
                    seed: $('#mc-seed').value || null,
                });
            } else {
                res = await API.mcPi({
                    n: $('#mc-n').value,
                    seed: $('#mc-seed').value || null,
                });
            }

            const area = $('#mc-resultado');
            area.style.display = '';

            const label = modo === 'integral' ? 'Estimación' : 'π ≈';
            showMessage($('#mc-msg'), res.mensaje, true,
                `${label} ${res.estimacion.toFixed(10)}  |  IC ${(res.confianza*100).toFixed(0)}%: [${res.ic_bajo.toFixed(6)}, ${res.ic_alto.toFixed(6)}]`);

            // Histogram with KDE-like smooth overlay
            Plotly.newPlot('mc-chart', [{
                x: res.aportes,
                type: 'histogram',
                nbinsx: 50,
                marker: {
                    color: 'rgba(139, 92, 246, 0.45)',
                    line: { color: 'rgba(139, 92, 246, 0.7)', width: 0.5 },
                },
                name: 'Distribución de aportes',
            }], plotlyLayout({
                title: { text: 'Distribución de Aportes Monte Carlo', font: { size: 15, color: '#f0f0f5' } },
                xaxis: { title: { text: 'Aporte' } },
                yaxis: { title: { text: 'Frecuencia' } },
                shapes: [{
                    type: 'line', x0: res.estimacion, x1: res.estimacion, y0: 0, y1: 1, yref: 'paper',
                    line: { color: '#22c55e', width: 2.5, dash: 'dash' },
                }, {
                    type: 'line', x0: res.ic_bajo, x1: res.ic_bajo, y0: 0, y1: 1, yref: 'paper',
                    line: { color: '#f59e0b', width: 1.5, dash: 'dot' },
                }, {
                    type: 'line', x0: res.ic_alto, x1: res.ic_alto, y0: 0, y1: 1, yref: 'paper',
                    line: { color: '#f59e0b', width: 1.5, dash: 'dot' },
                }],
                annotations: [{
                    x: res.estimacion, y: 1.03, yref: 'paper',
                    text: `μ = ${res.estimacion.toFixed(4)}`, showarrow: false,
                    font: { color: '#22c55e', size: 11 },
                }],
            }), PLOTLY_CFG);

            toast('Monte Carlo completado ✓', 'success');
        } catch (err) {
            toast(err.message, 'error');
            const area = $('#mc-resultado');
            area.style.display = '';
            showMessage($('#mc-msg'), err.message, false);
        } finally {
            setLoading(btn, false);
            $('#status-text').textContent = 'Listo';
        }
    });

    // ==================== EDO ====================

    on('#btn-edo', 'click', async () => {
        const btn = $('#btn-edo');
        if (!btn) return;
        const metodo = $('#edo-metodo').value;

        try {
            const payload = {
                ode_expr: $('#edo-expr').value,
                solucion_exacta: $('#edo-exacta').value.trim() || null,
                t0: $('#edo-t0').value,
                y0: $('#edo-y0').value,
                h: $('#edo-h').value,
                pasos: $('#edo-pasos').value,
            };

            setLoading(btn, true);
            $('#status-text').textContent = 'Resolviendo…';

            const apiFn = { euler: API.euler, heun: API.heun, rk4: API.rk4 }[metodo];
            const res = await apiFn(payload);
            const area = $('#edo-resultado');
            area.style.display = '';

            const lastStep = res.pasos[res.pasos.length - 1];
            showMessage($('#edo-msg'), res.mensaje, true,
                `y(${lastStep.t.toFixed(2)}) ≈ ${lastStep.y.toFixed(10)}`);

            // Table
            const hasExact = res.pasos.some(p => p.y_exacto != null);
            const cols = hasExact ? ['paso', 't', 'y', 'y_exacto', 'error_absoluto'] : ['paso', 't', 'y'];
            $('#edo-tabla').innerHTML = renderTable(res.pasos, cols);

            // Chart: trajectories
            const traces = [{
                x: res.pasos.map(p => p.t),
                y: res.pasos.map(p => p.y),
                mode: 'lines+markers',
                name: 'Aproximación numérica',
                line: { color: '#06b6d4', width: 2.5, shape: 'spline' },
                marker: { size: 5, color: '#06b6d4' },
            }];
            if (hasExact) {
                traces.push({
                    x: res.pasos.map(p => p.t),
                    y: res.pasos.map(p => p.y_exacto),
                    mode: 'lines',
                    name: 'Solución exacta',
                    line: { color: '#22c55e', width: 2, dash: 'dash' },
                });
                // Error trace (secondary y-axis)
                traces.push({
                    x: res.pasos.map(p => p.t),
                    y: res.pasos.map(p => p.error_absoluto),
                    mode: 'lines+markers',
                    name: 'Error absoluto',
                    line: { color: '#f59e0b', width: 1.5 },
                    marker: { size: 4, color: '#f59e0b' },
                    yaxis: 'y2',
                    opacity: 0.7,
                });
            }

            const layout = plotlyLayout({
                title: { text: `Trayectoria – ${metodo.toUpperCase()}`, font: { size: 15, color: '#f0f0f5' } },
                xaxis: { title: { text: 't' } },
                yaxis: { title: { text: 'y(t)' } },
            });
            if (hasExact) {
                layout.yaxis2 = {
                    title: { text: 'Error', font: { size: 11, color: '#f59e0b' } },
                    overlaying: 'y', side: 'right',
                    gridcolor: 'rgba(245,158,11,0.05)',
                    tickfont: { size: 10, color: '#f59e0b' },
                };
            }

            Plotly.newPlot('edo-chart', traces, layout, PLOTLY_CFG);

            toast('EDO resuelta ✓', 'success');
        } catch (err) {
            toast(err.message, 'error');
            const area = $('#edo-resultado');
            area.style.display = '';
            showMessage($('#edo-msg'), err.message, false);
        } finally {
            setLoading(btn, false);
            $('#status-text').textContent = 'Listo';
        }
    });

    // ==================== CASO PRÁCTICO INTEGRADO ====================

    const CASE_CHARTS = {};

    function caseAppNum(res, ...keys) {
        const app = res?.aplicacion || {};
        for (const key of keys) {
            const value = app[key];
            if (value !== undefined && value !== null && Number.isFinite(Number(value))) {
                return Number(value);
            }
        }
        return Number.NaN;
    }

    function initCaseChart(id) {
        const el = document.getElementById(id);
        if (!el || typeof echarts === 'undefined') return null;
        if (CASE_CHARTS[id]) {
            CASE_CHARTS[id].dispose();
        }
        CASE_CHARTS[id] = echarts.init(el, null, { renderer: 'canvas' });
        return CASE_CHARTS[id];
    }

    function buildCaseKpis(res) {
        const rf = caseAppNum(res, 'radio_hornalla_m');
        const rs = caseAppNum(res, 'radio_superficie_m', 'radio_disco_m');
        const areaRatio = rs > 0 ? (rf * rf) / (rs * rs) : 0;
        const cobertura = Math.max(0, Math.min(100, areaRatio * 100));
        const potencia = caseAppNum(res, 'potencia_hornalla_kw', 'potencia_hornalla');
        const tempSegura = caseAppNum(res, 'temperatura_segura_c');
        const kpis = [
            {
                title: 'Zona segura',
                value: `${res.aplicacion.distancia_segura_m.toFixed(3)} m`,
                note: `Meta: <= ${tempSegura.toFixed(0)} C`,
            },
            {
                title: 'Potencia',
                value: `${potencia.toFixed(2)} kW`,
                note: 'fuerza de la hornalla',
            },
            {
                title: 'Cobertura',
                value: `${cobertura.toFixed(1)}%`,
                note: 'área hornalla/superficie',
            },
            {
                title: 'Sartén final',
                value: res.edo.metodos.rk4.y_final.toFixed(3),
                note: 'EDO RK4',
            },
            {
                title: 'Latencia total',
                value: `${res.caso.runtime_ms.toFixed(1)} ms`,
                note: 'corrida integrada',
            },
        ];
        $('#caso-kpis').innerHTML = kpis.map(k => `
            <div class="kpi-card">
                <p class="kpi-title">${k.title}</p>
                <p class="kpi-value">${k.value}</p>
                <p class="kpi-note">${k.note}</p>
            </div>
        `).join('');
    }

    function buildSimpleCaseStory(res) {
        const el = $('#caso-historia');
        if (!el) return;
        const amb = caseAppNum(res, 'ambiente_c');
        const seg = caseAppNum(res, 'temperatura_segura_c');
        const safe = res.aplicacion.distancia_segura_m;
        const finalT = res.edo.metodos.rk4.y_final;
        const tempFuente = caseAppNum(res, 'temp_fuente_c', 'temp_objetivo_sarten_c');
        const finalDelta = Math.abs(finalT - tempFuente);
        const grad = res.interpolacion.derivada_t3;
        const ritmo = grad < -40 ? 'rápido' : grad < -20 ? 'moderado' : 'suave';

        el.innerHTML = `
            <div class="caso-simple-grid">
                <div class="caso-simple-card">
                    <h4>Lectura rápida</h4>
                    <p>
                        El ambiente está en <strong>${amb.toFixed(1)} °C</strong> y tomamos como límite seguro
                        <strong>${seg.toFixed(1)} °C</strong>.
                        La zona segura empieza aproximadamente a <strong>${safe.toFixed(2)} m</strong> del centro.
                    </p>
                </div>
                <div class="caso-simple-card">
                    <h4>¿Qué muestran los gráficos?</h4>
                    <p>
                        Cerca del centro el calor es más alto, y baja al alejarnos.
                        En este escenario, la caída de temperatura es <strong>${ritmo}</strong>
                        (gradiente: <strong>${grad.toFixed(2)} °C/m</strong>).
                    </p>
                </div>
                <div class="caso-simple-card">
                    <h4>Resumen temporal</h4>
                    <p>
                        La sartén termina en <strong>${finalT.toFixed(1)} °C</strong>.
                        Queda a <strong>${finalDelta.toFixed(1)} °C</strong> del valor máximo de la fuente.
                    </p>
                </div>
            </div>
        `;
    }

    function buildScenarioBadges(res) {
        const el = $('#caso-badges');
        if (!el) return;
        const buildBadge = (label, value) => `<span class="caso-badge"><strong>${label}:</strong> ${value}</span>`;
        const radioSuperficie = caseAppNum(res, 'radio_superficie_m', 'radio_disco_m');
        const tempFuente = caseAppNum(res, 'temp_fuente_c', 'temp_objetivo_sarten_c');
        el.innerHTML = [
            buildBadge('Radio hornalla', `${res.aplicacion.radio_hornalla_m.toFixed(2)} m`),
            buildBadge('Superficie analizada', `${radioSuperficie.toFixed(2)} m`),
            buildBadge('Temperatura fuente', `${tempFuente.toFixed(1)} °C`),
            buildBadge('Ambiente', `${res.aplicacion.ambiente_c.toFixed(1)} °C`),
            buildBadge('Intensidad', res.caso?.intensidad ?? 'pro'),
        ].join('');
    }

    function renderCaseCharts(res) {
        const chartRaices = initCaseChart('caso-chart-raices');
        const chartInteg = initCaseChart('caso-chart-integracion');
        const chartEdo = initCaseChart('caso-chart-edo');
        const chartMc = initCaseChart('caso-chart-mc');
        const chartAnim = initCaseChart('caso-chart-anim');

        if (!chartRaices || !chartInteg || !chartEdo || !chartMc || !chartAnim) {
            throw new Error('No se pudieron inicializar los charts ECharts.');
        }

        const surfaceRadius = caseAppNum(res, 'radio_superficie_m', 'radio_disco_m');
        chartRaices.setOption({
            backgroundColor: 'transparent',
            tooltip: {
                trigger: 'item',
                backgroundColor: 'rgba(10,10,20,0.9)',
                borderColor: 'rgba(6,182,212,0.3)',
                textStyle: { color: '#e2e8f0', fontSize: 12 },
                formatter: (p) => `<strong style="color:#67e8f9">Punto</strong><br/>x = ${p.value[0].toFixed(3)} m<br/>y = ${p.value[1].toFixed(3)} m<br/>T = ${p.value[2].toFixed(1)} °C`,
            },
            grid: { left: 56, right: 24, top: 28, bottom: 42 },
            xAxis: {
                type: 'value',
                name: 'x (m)',
                nameTextStyle: { color: '#94a3b8', fontSize: 11 },
                min: -surfaceRadius,
                max: surfaceRadius,
                axisLine: { lineStyle: { color: '#334155' } },
                axisLabel: { color: '#94a3b8' },
                splitLine: { lineStyle: { color: 'rgba(148,163,184,0.06)' } },
            },
            yAxis: {
                type: 'value',
                name: 'y (m)',
                nameTextStyle: { color: '#94a3b8', fontSize: 11 },
                min: -surfaceRadius,
                max: surfaceRadius,
                axisLine: { lineStyle: { color: '#334155' } },
                splitLine: { lineStyle: { color: 'rgba(148,163,184,0.08)' } },
                axisLabel: { color: '#94a3b8' },
            },
            visualMap: {
                min: res.aplicacion.ambiente_c,
                max: Math.max(...res.visualizacion.nube_puntos.map(p => p[2])),
                dimension: 2,
                calculable: true,
                orient: 'vertical',
                right: 8,
                top: 'middle',
                textStyle: { color: '#cbd5e1' },
                inRange: { color: ['#1e3a5f', '#0d9488', '#22c55e', '#f59e0b', '#ef4444', '#fbbf24'] },
            },
            series: [{
                name: 'Mapa térmico',
                type: 'heatmap',
                data: res.visualizacion.heatmap_estatico,
                blurSize: 18,
                pointSize: 5,
                itemStyle: { opacity: 0.55 },
                z: 1,
            }, {
                name: 'Temperatura',
                type: 'scatter',
                data: res.visualizacion.nube_puntos,
                symbolSize: 7,
                itemStyle: { opacity: 0.85 },
                emphasis: { scale: 1.8, itemStyle: { shadowBlur: 10, shadowColor: 'rgba(6,182,212,0.5)' } },
                z: 2,
            }, {
                name: 'Centro hornalla',
                type: 'scatter',
                data: [[0, 0, 0]],
                symbolSize: 16,
                symbol: 'diamond',
                itemStyle: { color: '#fbbf24', borderColor: '#0f172a', borderWidth: 2 },
                z: 3,
            }],
        });

        // ---- Perfil Radial + Sensores ----
        const rSafe = res.aplicacion.distancia_segura_m;
        const rHornalla = res.aplicacion.radio_hornalla_m;
        chartInteg.setOption({
            backgroundColor: 'transparent',
            tooltip: {
                trigger: 'axis',
                backgroundColor: 'rgba(10,10,20,0.9)',
                borderColor: 'rgba(6,182,212,0.3)',
                textStyle: { color: '#e2e8f0', fontSize: 12 },
            },
            legend: { textStyle: { color: '#9ca3af' }, top: 4 },
            grid: { left: 60, right: 22, top: 52, bottom: 48 },
            xAxis: {
                type: 'value',
                name: 'Distancia radial (m)',
                nameTextStyle: { color: '#94a3b8', fontSize: 11 },
                axisLabel: { color: '#94a3b8' },
                axisLine: { lineStyle: { color: '#334155' } },
                splitLine: { lineStyle: { color: 'rgba(148,163,184,0.06)' } },
            },
            yAxis: {
                type: 'value',
                name: 'Temperatura (°C)',
                nameTextStyle: { color: '#94a3b8', fontSize: 11 },
                axisLabel: { color: '#94a3b8' },
                splitLine: { lineStyle: { color: 'rgba(148,163,184,0.08)' } },
            },
            series: [{
                name: 'Modelo térmico',
                type: 'line',
                smooth: true,
                data: res.visualizacion.curva_radial.r.map((r, i) => [r, res.visualizacion.curva_radial.temp[i]]),
                lineStyle: { width: 3, color: '#22d3ee' },
                areaStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: 'rgba(34,211,238,0.25)' },
                        { offset: 1, color: 'rgba(34,211,238,0.01)' },
                    ]),
                },
                showSymbol: false,
            }, {
                name: 'Sensores',
                type: 'scatter',
                data: res.interpolacion.puntos,
                symbolSize: 13,
                itemStyle: {
                    color: '#f59e0b',
                    borderColor: '#fff',
                    borderWidth: 1.5,
                },
                emphasis: { scale: 1.6 },
            }],
            markLine: {
                silent: true,
                data: [
                    { xAxis: rHornalla, lineStyle: { color: '#f59e0b', type: 'solid', width: 2 } },
                    { xAxis: rSafe, lineStyle: { color: '#22c55e', type: 'dashed', width: 2 } },
                ],
            },
        });

        // ---- Potencia Térmica Integrada (barras coloreadas) ----
        const integLabels = ['Trapecio', 'Simpson 1/3', 'Simpson 3/8', 'Rectángulo', 'Gauss', 'MC'];
        const integValues = [
            res.integracion.resultados.trapecio,
            res.integracion.resultados.simpson13,
            res.integracion.resultados.simpson38,
            res.integracion.resultados.rectangulo,
            res.integracion.resultados.gauss_legendre,
            res.montecarlo.integral.estimacion,
        ];
        const barColors = ['#06b6d4', '#8b5cf6', '#a78bfa', '#22c55e', '#f59e0b', '#ec4899'];

        chartEdo.setOption({
            backgroundColor: 'transparent',
            tooltip: {
                trigger: 'axis',
                backgroundColor: 'rgba(10,10,20,0.9)',
                borderColor: 'rgba(6,182,212,0.3)',
                textStyle: { color: '#e2e8f0', fontSize: 12 },
                formatter: (params) => {
                    const p = params[0];
                    return `<strong style="color:${barColors[p.dataIndex]}">${p.name}</strong><br/>Valor: ${p.value.toFixed(6)}`;
                },
            },
            grid: { left: 60, right: 20, top: 28, bottom: 56 },
            xAxis: {
                type: 'category',
                data: integLabels,
                axisLabel: { color: '#94a3b8', fontSize: 11, rotate: 15 },
                axisLine: { lineStyle: { color: '#334155' } },
            },
            yAxis: {
                type: 'value',
                name: 'Potencia acumulada',
                nameTextStyle: { color: '#94a3b8', fontSize: 11 },
                axisLabel: { color: '#94a3b8' },
                splitLine: { lineStyle: { color: 'rgba(148,163,184,0.08)' } },
            },
            series: [{
                type: 'bar',
                data: integValues.map((v, i) => ({
                    value: v,
                    itemStyle: {
                        borderRadius: [8, 8, 0, 0],
                        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            { offset: 0, color: barColors[i] },
                            { offset: 1, color: barColors[i] + '55' },
                        ]),
                    },
                })),
                barWidth: '48%',
                label: {
                    show: true,
                    position: 'top',
                    color: '#cbd5e1',
                    fontSize: 10,
                    fontFamily: 'JetBrains Mono, monospace',
                    formatter: (p) => p.value.toFixed(3),
                },
            }],
        });

        // ---- Calentamiento de Sartén (EDO) ----
        chartMc.setOption({
            backgroundColor: 'transparent',
            tooltip: {
                trigger: 'axis',
                backgroundColor: 'rgba(10,10,20,0.9)',
                borderColor: 'rgba(6,182,212,0.3)',
                textStyle: { color: '#e2e8f0', fontSize: 12 },
            },
            legend: {
                textStyle: { color: '#9ca3af' },
                top: 4,
                selectedMode: 'multiple',
            },
            grid: { left: 60, right: 22, top: 48, bottom: 42 },
            xAxis: {
                type: 'category',
                data: res.edo.trayectorias.t,
                name: 'Tiempo (s)',
                nameTextStyle: { color: '#94a3b8', fontSize: 11 },
                axisLabel: { color: '#94a3b8' },
                axisLine: { lineStyle: { color: '#334155' } },
                splitLine: { lineStyle: { color: 'rgba(148,163,184,0.06)' } },
            },
            yAxis: {
                type: 'value',
                name: 'Temperatura sartén (°C)',
                nameTextStyle: { color: '#94a3b8', fontSize: 11 },
                axisLabel: { color: '#94a3b8' },
                splitLine: { lineStyle: { color: 'rgba(148,163,184,0.08)' } },
            },
            series: [
                { name: 'Euler', type: 'line', smooth: true, data: res.edo.trayectorias.euler, lineStyle: { width: 2.5, color: '#06b6d4' }, symbol: 'circle', symbolSize: 4 },
                { name: 'Heun', type: 'line', smooth: true, data: res.edo.trayectorias.heun, lineStyle: { width: 2.5, color: '#8b5cf6' }, symbol: 'rect', symbolSize: 4 },
                { name: 'RK4', type: 'line', smooth: true, data: res.edo.trayectorias.rk4, lineStyle: { width: 3, color: '#22c55e' }, symbol: 'triangle', symbolSize: 5 },
                { name: 'Exacta', type: 'line', smooth: true, data: res.edo.trayectorias.exacta, lineStyle: { width: 2, color: '#f59e0b', type: 'dashed' }, symbol: 'none' },
            ],
        });

        // ---- Mapa Térmico Animado ----
        const frames = res.visualizacion.heatmap_animado.frames || [];
        const maxTemp = Math.max(...res.visualizacion.heatmap_estatico.map(p => p[2]));
        chartAnim.setOption({
            baseOption: {
                backgroundColor: 'transparent',
                timeline: {
                    axisType: 'category',
                    autoPlay: true,
                    playInterval: 400,
                    data: frames.map(f => `${f.t.toFixed(1)}s`),
                    label: { color: '#94a3b8', fontSize: 11 },
                    lineStyle: { color: '#334155' },
                    controlStyle: { color: '#cbd5e1', borderColor: '#475569' },
                    checkpointStyle: {
                        color: '#06b6d4',
                        borderColor: '#0891b2',
                        animation: true,
                    },
                    emphasis: { controlStyle: { color: '#22d3ee' } },
                    bottom: 6,
                },
                tooltip: {
                    trigger: 'item',
                    backgroundColor: 'rgba(10,10,20,0.9)',
                    borderColor: 'rgba(6,182,212,0.3)',
                    textStyle: { color: '#e2e8f0', fontSize: 12 },
                    formatter: (p) => `<strong style="color:#67e8f9">Celda</strong><br/>x = ${p.value[0].toFixed(3)} m<br/>y = ${p.value[1].toFixed(3)} m<br/>T = ${p.value[2].toFixed(1)} °C`,
                },
                visualMap: {
                    min: res.aplicacion.ambiente_c,
                    max: maxTemp,
                    dimension: 2,
                    orient: 'vertical',
                    right: 8,
                    top: 'middle',
                    calculable: true,
                    textStyle: { color: '#cbd5e1' },
                    inRange: { color: ['#1e3a5f', '#0d9488', '#22c55e', '#f59e0b', '#ef4444', '#fbbf24'] },
                },
                grid: { left: 56, right: 24, top: 32, bottom: 72 },
                xAxis: {
                    type: 'value',
                    name: 'x (m)',
                    nameTextStyle: { color: '#94a3b8', fontSize: 11 },
                    min: -surfaceRadius,
                    max: surfaceRadius,
                    axisLine: { lineStyle: { color: '#334155' } },
                    axisLabel: { color: '#94a3b8' },
                    splitLine: { lineStyle: { color: 'rgba(148,163,184,0.06)' } },
                },
                yAxis: {
                    type: 'value',
                    name: 'y (m)',
                    nameTextStyle: { color: '#94a3b8', fontSize: 11 },
                    min: -surfaceRadius,
                    max: surfaceRadius,
                    axisLine: { lineStyle: { color: '#334155' } },
                    splitLine: { lineStyle: { color: 'rgba(148,163,184,0.06)' } },
                    axisLabel: { color: '#94a3b8' },
                },
            },
            options: frames.map(f => ({
                title: {
                    text: `Propagación Térmica · t = ${f.t.toFixed(1)} s`,
                    left: 'center',
                    top: 4,
                    textStyle: { color: '#e2e8f0', fontSize: 13, fontWeight: 600, fontFamily: 'Inter, sans-serif' },
                },
                series: [{
                    type: 'heatmap',
                    data: f.data,
                    blurSize: 18,
                    pointSize: 5,
                }],
            })),
        });

        window.requestAnimationFrame(() => {
            Object.values(CASE_CHARTS).forEach(ch => ch && ch.resize());
        });
    }

    // --- Generador de conclusiones interpretativas ---
    function buildCaseConclusions(res) {
        const el = $('#caso-conclusiones');
        if (!el) return;

        const safe   = res.aplicacion.distancia_segura_m;
        const safeT  = caseAppNum(res, 'temperatura_segura_c');
        const nwIter = res.raices.newton_raphson.iteraciones;
        const biIter = res.raices.biseccion.iteraciones;
        const aiIter = res.raices.aitken.iteraciones;

        const s13  = res.integracion.resultados.simpson13;
        const trap = res.integracion.resultados.trapecio;
        const mcI  = res.montecarlo.integral.estimacion;
        const mcLo = res.montecarlo.integral.ic_bajo;
        const mcHi = res.montecarlo.integral.ic_alto;

        const rk4F = res.edo.metodos.rk4.y_final;
        const euF  = res.edo.metodos.euler?.y_final ?? rk4F;
        const rk4E = res.edo.metodos.rk4.error_final;

        const piEst = res.montecarlo.pi.estimacion;
        const piErr = Math.abs(piEst - Math.PI);

        const intDiff = Math.abs(s13 - mcI);
        const intPct  = s13 !== 0 ? ((intDiff / Math.abs(s13)) * 100).toFixed(2) : '—';
        const rHornalla = caseAppNum(res, 'radio_hornalla_m');
        const tFuente = caseAppNum(res, 'temp_fuente_c', 'temp_objetivo_sarten_c');
        const potencia = caseAppNum(res, 'potencia_hornalla_kw', 'potencia_hornalla');

        const items = [
            {
                icon: '🎯',
                title: 'Zona de seguridad',
                text: `Con radio de hornalla <strong>${rHornalla.toFixed(2)} m</strong>, potencia <strong>${potencia.toFixed(2)} kW</strong> y temperatura de fuente <strong>${tFuente.toFixed(0)} °C</strong>, la zona segura se ubica alrededor de <strong>${safe.toFixed(3)} m</strong> (meta: T <= ${safeT.toFixed(0)} °C).`,
            },
            {
                icon: '📏',
                title: 'Interpolación y gradiente',
                text: `Con pocos sensores se reconstruyó el perfil completo de calor. El gradiente en r = 0.30 m fue <strong>${res.interpolacion.derivada_t3.toFixed(3)} °C/m</strong>: cuanto más negativo, más rápido cae la temperatura al alejarse.`,
            },
            {
                icon: '⚡',
                title: 'Potencia térmica integrada',
                text: `Los métodos dieron valores parecidos (Simpson 1/3 = <strong>${s13.toFixed(4)}</strong>, Trapecio = <strong>${trap.toFixed(4)}</strong>). En la práctica, eso indica que la estimación de energía es estable.`,
            },
            {
                icon: '🎲',
                title: 'Monte Carlo vs. determinístico',
                text: `Monte Carlo estimó <strong>${mcI.toFixed(4)}</strong> con rango probable [${mcLo.toFixed(3)}, ${mcHi.toFixed(3)}]. La diferencia frente a Simpson fue de <strong>${intPct}%</strong>, o sea que ambos cuentan una historia parecida.`,
            },
            {
                icon: '🔥',
                title: 'Calentamiento de sartén (EDO)',
                text: `RK4 terminó en <strong>${rk4F.toFixed(2)} °C</strong> con error muy chico (<strong>${rk4E.toExponential(2)}</strong>). Euler quedó en <strong>${euF.toFixed(2)} °C</strong>, mostrando una diferencia mayor.`,
            },
            {
                icon: '🥧',
                title: 'Estimación de π',
                text: `Como control de calidad del muestreo, Monte Carlo estimó π ≈ <strong>${piEst.toFixed(6)}</strong> con error <strong>${piErr.toFixed(5)}</strong>.`,
            },
        ];

        let html = items.map(it => `
            <div class="conclusion-item">
                <span class="conclusion-icon">${it.icon}</span>
                <div class="conclusion-body">
                    <p class="conclusion-title">${it.title}</p>
                    <p class="conclusion-text">${it.text}</p>
                </div>
            </div>
        `).join('');

        // Veredicto final
        html += `
            <div class="conclusion-veredicto">
                <span class="veredicto-label">VEREDICTO FINAL</span>
                <p class="veredicto-text">
                    En simple: cuando subís potencia o temperatura de fuente, el calor se expande más;
                    si achicás radio de hornalla o potencia, se concentra menos.
                    Lo importante es que el panel te deja ver rápido
                    <strong>dónde está la zona segura y cómo cambia al mover parámetros</strong>.
                </p>
            </div>
        `;

        el.innerHTML = html;
    }

    window.addEventListener('resize', () => {
        Object.values(CASE_CHARTS).forEach(ch => ch && ch.resize());
    });

    function getCaseInputValue(id, fallback = '') {
        const el = document.getElementById(id);
        if (!el) return fallback;
        return (el.value ?? '').toString();
    }

    function getCaseInputNumber(id, fallback = NaN) {
        const raw = getCaseInputValue(id, '');
        if (raw === '') return fallback;
        const num = Number.parseFloat(raw);
        return Number.isFinite(num) ? num : fallback;
    }

    on('#btn-caso-practico', 'click', async () => {
        const btn = $('#btn-caso-practico');
        if (!btn) return;
        try {
            setLoading(btn, true);
            $('#status-text').textContent = 'Ejecutando escenario…';

            const intensidad = getCaseInputValue('caso-intensidad', 'pro');
            const seed = getCaseInputValue('caso-seed', '42').trim() || '42';
            const radioHornalla = getCaseInputNumber('caso-radio-hornalla', 0.11);
            const ambiente = getCaseInputNumber('caso-ambiente', 24.0);
            const tempSegura = getCaseInputNumber('caso-temperatura-segura', 60.0);
            const potencia = getCaseInputNumber('caso-potencia', 320.0);
            const alpha = getCaseInputNumber('caso-alpha', 3.2);
            const perdidaAire = getCaseInputNumber('caso-perdida-aire', 10.0);
            const ruidoSensor = getCaseInputNumber('caso-ruido-sensor', 4.0);

            const invalid = [radioHornalla, ambiente, tempSegura, potencia, alpha, perdidaAire, ruidoSensor]
                .some(v => !Number.isFinite(v));
            if (invalid) {
                throw new Error('Revisá los parámetros: hay un valor vacío o inválido.');
            }

            const profile = {
                base: { mc_n: 6000, pi_n: 8000, cloud_n: 900, pasos: 12, h: 0.5 },
                pro: { mc_n: 20000, pi_n: 25000, cloud_n: 1800, pasos: 20, h: 0.3 },
                extremo: { mc_n: 50000, pi_n: 60000, cloud_n: 3000, pasos: 30, h: 0.2 },
            }[intensidad] || { mc_n: 20000, pi_n: 25000, cloud_n: 1800, pasos: 20, h: 0.3 };

            const res = await API.casoPracticoIntegrado({
                ...profile,
                seed,
                radio_hornalla_m: radioHornalla,
                ambiente_c: ambiente,
                temperatura_segura_c: tempSegura,
                potencia_hornalla: potencia,
                rapidez_disipacion: alpha,
                perdida_lineal: perdidaAire,
                ruido_sensor: ruidoSensor,
                intensidad,
            });
            const area = $('#caso-resultado');
            area.style.display = '';

            showMessage(
                $('#caso-msg'),
                res.mensaje,
                true,
                `${res.caso.titulo} · ${res.caso.descripcion}`
            );

            buildCaseKpis(res);
            buildScenarioBadges(res);
            buildSimpleCaseStory(res);

            const resumenRows = [
                { bloque: 'Raíces · Distancia segura (Newton)', valor: res.raices.newton_raphson.aproximacion, detalle: `iter=${res.raices.newton_raphson.iteraciones}` },
                { bloque: 'Raíces · Bisección', valor: res.raices.biseccion.aproximacion, detalle: `iter=${res.raices.biseccion.iteraciones}` },
                { bloque: 'Raíces · Aitken', valor: res.raices.aitken.aproximacion, detalle: `iter=${res.raices.aitken.iteraciones}` },
                { bloque: 'Interpolación · T(r=0.38m)', valor: res.interpolacion.valor_interpolado, detalle: 'lectura intermedia' },
                { bloque: 'Gradiente radial · dT/dr (r=0.30m)', valor: res.interpolacion.derivada_t3, detalle: 'diferencia central' },
                { bloque: 'Integración · Simpson 1/3', valor: res.integracion.resultados.simpson13, detalle: `[0, ${caseAppNum(res, 'radio_superficie_m', 'radio_disco_m').toFixed(2)}] m` },
                { bloque: 'Monte Carlo · Integral', valor: res.montecarlo.integral.estimacion, detalle: `IC95% [${res.montecarlo.integral.ic_bajo.toFixed(3)}, ${res.montecarlo.integral.ic_alto.toFixed(3)}]` },
                { bloque: 'Monte Carlo · Pi', valor: res.montecarlo.pi.estimacion, detalle: `IC95% [${res.montecarlo.pi.ic_bajo.toFixed(3)}, ${res.montecarlo.pi.ic_alto.toFixed(3)}]` },
                { bloque: 'EDO · Sartén final RK4', valor: res.edo.metodos.rk4.y_final, detalle: `error=${res.edo.metodos.rk4.error_final.toExponential(3)}` },
            ];
            $('#caso-resumen').innerHTML = renderTable(resumenRows, ['bloque', 'valor', 'detalle']);

            renderCaseCharts(res);
            buildCaseConclusions(res);

            toast('Caso práctico ejecutado ✓', 'success');
        } catch (err) {
            toast(err.message, 'error');
            const area = $('#caso-resultado');
            area.style.display = '';
            showMessage($('#caso-msg'), err.message, false);
        } finally {
            setLoading(btn, false);
            $('#status-text').textContent = 'Listo';
        }
    });



    // ==================== Init ====================

    TheoryPanel.update('raices');

    window.addEventListener('load', () => {
        if (typeof renderMathInElement === 'function') {
            TheoryPanel.update('raices');
        }
    });

})();
