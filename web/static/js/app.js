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

    $('#menu-toggle').addEventListener('click', () => {
        $('#sidebar').classList.toggle('open');
    });

    // ==================== RAÍCES ====================

    const raicesMetodo = $('#raices-metodo');
    raicesMetodo.addEventListener('change', () => {
        const m = raicesMetodo.value;
        $$('.method-fields', $('#sec-raices')).forEach(f => f.style.display = 'none');
        const target = $(`#raices-fields-${m}`);
        if (target) target.style.display = '';
    });

    $('#btn-raices').addEventListener('click', async () => {
        const btn = $('#btn-raices');
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
        Array.from(pointsList.children).forEach((row, i) => {
            row.querySelector('.point-label').textContent = `P${i + 1}`;
        });
    }

    function getPointsFromEditor() {
        return Array.from(pointsList.querySelectorAll('.point-row')).map(row => {
            const x = parseFloat(row.querySelector('.interp-x').value);
            const y = parseFloat(row.querySelector('.interp-y').value);
            return [x, y];
        }).filter(p => !isNaN(p[0]) && !isNaN(p[1]));
    }

    // Init default points
    defaultPoints.forEach(([x, y]) => addPointRow(x, y));

    $('#interp-add-point').addEventListener('click', () => addPointRow());

    $('#btn-interp').addEventListener('click', async () => {
        const btn = $('#btn-interp');
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

    $('#btn-deriv').addEventListener('click', async () => {
        const btn = $('#btn-deriv');
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

    $('#btn-integ').addEventListener('click', async () => {
        const btn = $('#btn-integ');
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
    mcModo.addEventListener('change', () => {
        $('#mc-fields-integral').style.display = mcModo.value === 'integral' ? '' : 'none';
    });

    $('#btn-mc').addEventListener('click', async () => {
        const btn = $('#btn-mc');
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

    $('#btn-edo').addEventListener('click', async () => {
        const btn = $('#btn-edo');
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



    // ==================== Init ====================

    TheoryPanel.update('raices');

    window.addEventListener('load', () => {
        if (typeof renderMathInElement === 'function') {
            TheoryPanel.update('raices');
        }
    });

})();
