/**
 * api.js – Módulo de comunicación con el backend Flask.
 */

const API = (() => {
    const BASE = '';

    async function fetchJSON(endpoint, data) {
        const resp = await fetch(`${BASE}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        const json = await resp.json();
        if (!resp.ok) {
            throw new Error(json.error || `Error ${resp.status}`);
        }
        return json;
    }

    return {
        // Raíces
        biseccion:      (d) => fetchJSON('/api/raices/biseccion', d),
        puntoFijo:      (d) => fetchJSON('/api/raices/punto-fijo', d),
        newtonRaphson:  (d) => fetchJSON('/api/raices/newton-raphson', d),
        aitken:         (d) => fetchJSON('/api/raices/aitken', d),

        // Interpolación
        lagrange:       (d) => fetchJSON('/api/interpolacion/lagrange', d),
        derivadaCentral:(d) => fetchJSON('/api/interpolacion/derivada-central', d),

        // Integración
        trapecio:       (d) => fetchJSON('/api/integracion/trapecio', d),
        simpson13:      (d) => fetchJSON('/api/integracion/simpson13', d),
        simpson38:      (d) => fetchJSON('/api/integracion/simpson38', d),
        rectangulo:     (d) => fetchJSON('/api/integracion/rectangulo', d),
        gaussLegendre:  (d) => fetchJSON('/api/integracion/gauss-legendre', d),

        // Monte Carlo
        mcIntegral:     (d) => fetchJSON('/api/montecarlo/integral', d),
        mcPi:           (d) => fetchJSON('/api/montecarlo/pi', d),

        // EDO
        euler:          (d) => fetchJSON('/api/edo/euler', d),
        heun:           (d) => fetchJSON('/api/edo/heun', d),
        rk4:            (d) => fetchJSON('/api/edo/rk4', d),

        // Caso práctico integrado
        casoPracticoIntegrado: (d = {}) => fetchJSON('/api/casos/practico-integrado', d),

        // Sistemas lineales
        gauss:          (d) => fetchJSON('/api/sistemas/gauss', d),

        // Utilidad
        evaluarCurva:   (d) => fetchJSON('/api/util/evaluar-curva', d),
    };
})();
