// Ejecuta el camino `?job=` de la vista SIN navegador.
//
// Saca las funciones del propio vista.html (no una copia: se cortan del archivo
// por nombre y se evaluan) y reproduce cargarDesdeCore paso a paso. Asi un
// ReferenceError como el de areaPoly aparece aca, en dos segundos, en vez de
// aparecer como un cartel generico en la pantalla del usuario.

import { readFileSync } from "node:fs";

const RUTA = "D:/PROYECTOS/Enaex - Flyrocks/detovision_standalone/flyrocks_core/debug/demo/vista.html";
const API = "http://localhost:8009";
const JOB = process.argv[2];

const html = readFileSync(RUTA, "utf8");

// Corta una funcion por su nombre, contando llaves.
function extraer(nombre) {
  const i = html.indexOf(`function ${nombre}(`);
  if (i < 0) throw new Error(`no existe la funcion ${nombre} en vista.html`);
  let prof = 0, dentro = false;
  for (let j = i; j < html.length; j++) {
    if (html[j] === "{") { prof++; dentro = true; }
    else if (html[j] === "}") { prof--; if (dentro && prof === 0) return html.slice(i, j + 1); }
  }
  throw new Error(`no se pudo cerrar ${nombre}`);
}

const NECESARIAS = ["areaPoly", "aH3", "escalaDeH", "separacionMedia", "normalizarDelCore"];
const faltan = [];
for (const f of NECESARIAS) {
  try { extraer(f); } catch { faltan.push(f); }
}
if (faltan.length) {
  console.log("FALTAN FUNCIONES:", faltan.join(", "));
  process.exit(1);
}
console.log("funciones presentes:", NECESARIAS.join(", "));

const ctx = {};
for (const f of NECESARIAS) {
  // eslint-disable-next-line no-eval
  ctx[f] = (0, eval)(`(${extraer(f)})`);
}
const { areaPoly, aH3, escalaDeH, separacionMedia, normalizarDelCore } = ctx;

const r = await fetch(`${API}/api/results/${JOB}`);
const J = await r.json();
if (J.detail) { console.log("el core no conoce el job:", J.detail); process.exit(1); }
if (J.is_running) { console.log("todavia corriendo"); process.exit(1); }

const E = J.entrada || {};
const paso = (n, fn) => {
  try { const v = fn(); console.log(`  OK  ${n}${v !== undefined ? " -> " + v : ""}`); return v; }
  catch (e) { console.log(`  FALLA ${n}: ${e.constructor.name}: ${e.message}`); process.exit(1); }
};

console.log("\nreproduciendo cargarDesdeCore:");
paso("entrada.h_matrix presente", () => { if (!E.h_matrix) throw new Error("falta"); return "si"; });

const aPares = (z) => {
  if (!z || !z.length) return [];
  const p = z[0];
  return (Array.isArray(p) && p.length === 2 && z.every(q => q.length === 2))
    ? z.map(q => [+q[0], +q[1]])
    : Array.from({ length: Math.floor(p.length / 2) }, (_, i) => [+p[2 * i], +p[2 * i + 1]]);
};
const origen = paso("aPares(origin_zone)", () => aPares(E.origin_zone)).length !== undefined
  ? aPares(E.origin_zone) : [];
const seguridad = aPares(E.expected_projection_zone);
console.log(`      origen ${origen.length} vertices | seguridad ${seguridad.length}`);

const h3 = paso("aH3(h_matrix)", () => { const h = aH3(E.h_matrix); if (!h) throw new Error("no son 9 numeros"); return "3x3"; }) && aH3(E.h_matrix);
const escala = paso("escalaDeH", () => escalaDeH(h3).toFixed(4) + " px/m");
const esc = escalaDeH(h3);
const areaM2 = paso("areaPoly(origen)", () => (areaPoly(origen) / (esc * esc)).toFixed(0) + " m2");
const a = areaPoly(origen) / (esc * esc);
paso("diametro equivalente", () => (2 * Math.sqrt(Math.max(a, 1) / Math.PI)).toFixed(2) + " m");
paso("separacionMedia (radio seguridad)", () => (separacionMedia(origen, seguridad) / esc).toFixed(1) + " m");

const fps = (E.malla && E.malla.meta.fps) || 29.97;
const malla = paso("bloque malla", () => {
  const m = (E.malla && E.malla.pozos && E.malla.pozos.length)
    ? { meta: { ...E.malla.meta, frame_inicio: E.malla.meta.frame_inicio ?? 0, fps }, pozos: E.malla.pozos }
    : { meta: { t_min: 0, t_max: 1, frame_inicio: 0, fps, A: [[1, 0], [0, 1]], t: [0, 0] }, pozos: [] };
  return `${m.pozos.length} pozos, fps ${m.meta.fps}, ancla ${m.meta.frame_inicio}`;
});

const P = paso("normalizarDelCore", () => normalizarDelCore(J.json_data || {}).length + " trayectorias");
const proy = normalizarDelCore(J.json_data || {});
paso("con eje temporal", () => proy.filter(t => t.t_ini != null).length + " / " + proy.length);

console.log("\nLA VISTA PUEDE CARGAR ESTE JOB.");
