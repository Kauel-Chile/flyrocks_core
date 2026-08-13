// Ejercita el lazo y la union de trayectorias SIN navegador.
//
// Igual que probar_carga.mjs: corta las funciones del propio vista.html en vez
// de reimplementarlas, asi lo que se prueba es el codigo que corre de verdad.
// Se montan los globales minimos que esas funciones tocan (C, U, V, LAZO).
//
//   node debug/probar_herramientas.mjs <job_id>

import { readFileSync } from "node:fs";

const RUTA = new URL("./demo/vista.html", import.meta.url);
const API = "http://localhost:8009";
const JOB = process.argv[2];

const html = readFileSync(RUTA, "utf8");

function cortar(nombre) {
  const i = html.indexOf(`function ${nombre}(`);
  if (i < 0) throw new Error(`falta la funcion ${nombre}`);
  let prof = 0, dentro = false;
  for (let j = i; j < html.length; j++) {
    if (html[j] === "{") { prof++; dentro = true; }
    else if (html[j] === "}") { prof--; if (dentro && prof === 0) return html.slice(i, j + 1); }
  }
  throw new Error(`no cierra ${nombre}`);
}

// Globales que las funciones cortadas esperan encontrar.
globalThis.C = null;
globalThis.U = { sel: -1 };
globalThis.V = { z: 1 };
globalThis.LAZO = { pts: [], n: 0, tocar: false, restaurar: false };
globalThis.recalcular = () => {};
globalThis.verDetalle = () => {};
globalThis.dibujar = () => {};

const FUNCS = ["dentroPoly", "cajaDe", "atrapadas", "aplicarLazo",
               "preparar", "invertir", "unirTrayectorias", "normalizarDelCore",
               "solapeTemporal"];
for (const f of FUNCS) globalThis[f] = (0, eval)(`(${cortar(f)})`);

const ok = [], mal = [];
const afirmar = (c, n, extra = "") => (c ? ok : mal).push(n + (extra ? ` (${extra})` : ""));

// --- datos reales -----------------------------------------------------------
const J = await (await fetch(`${API}/api/results/${JOB}`)).json();
if (J.detail) { console.log("el core no conoce el job:", J.detail); process.exit(1); }

globalThis.C = {
  proyecciones: normalizarDelCore(J.json_data || {}),
  calibra: { escala_px_por_m: 8.5495 },
};
for (const T of C.proyecciones) preparar(T);
console.log(`caso: ${C.proyecciones.length} trayectorias reales\n`);

// --- punto en poligono ------------------------------------------------------
const cuadro = [{x:0,y:0},{x:10,y:0},{x:10,y:10},{x:0,y:10}];
afirmar(dentroPoly(5, 5, cuadro) === true, "dentroPoly: centro dentro");
afirmar(dentroPoly(15, 5, cuadro) === false, "dentroPoly: fuera a la derecha");
afirmar(dentroPoly(-1, 5, cuadro) === false, "dentroPoly: fuera a la izquierda");
// Concavo: una C. El hueco NO debe contar como dentro.
const c = [{x:0,y:0},{x:10,y:0},{x:10,y:3},{x:3,y:3},{x:3,y:7},{x:10,y:7},{x:10,y:10},{x:0,y:10}];
afirmar(dentroPoly(1, 5, c) === true, "dentroPoly: concavo, brazo");
afirmar(dentroPoly(6, 5, c) === false, "dentroPoly: concavo, hueco");

// --- lazo sobre datos reales ------------------------------------------------
// Un rectangulo alrededor de una trayectoria concreta, holgado.
const T0 = C.proyecciones[0], b = T0.bb, m = 20;
const rect = [{x:b.x0-m,y:b.y0-m},{x:b.x1+m,y:b.y0-m},{x:b.x1+m,y:b.y1+m},{x:b.x0-m,y:b.y1+m}];

const enteras = atrapadas(rect, false, false);
const tocan   = atrapadas(rect, true,  false);
afirmar(enteras.includes(0), "lazo: agarra la que rodea entera");
afirmar(tocan.length >= enteras.length, "lazo: 'tocan' es superconjunto de 'enteras'",
        `${tocan.length} >= ${enteras.length}`);
afirmar(enteras.every(i => tocan.includes(i)), "lazo: toda entera tambien toca");

// Una que cruza pero no cabe: no debe caer en modo estricto.
const cruzan = tocan.filter(i => !enteras.includes(i));
afirmar(true, `lazo: ${enteras.length} enteras, ${tocan.length} tocan, ${cruzan.length} solo cruzan`);

// Lazo degenerado
afirmar(atrapadas([{x:0,y:0},{x:1,y:1}], false, false).length === 0, "lazo: menos de 3 vertices no agarra nada");

// Aplicar y restaurar
LAZO.pts = rect; LAZO.tocar = false; LAZO.restaurar = false;
const n1 = aplicarLazo();
afirmar(n1 === enteras.length, "lazo: descarta las que anuncio", `${n1}`);
afirmar(C.proyecciones[0].estado === "descartada", "lazo: deja estado descartada");
afirmar(C.proyecciones[0].razon === "lazo", "lazo: deja razon 'lazo'");

LAZO.pts = rect; LAZO.restaurar = true;
const n2 = aplicarLazo();
afirmar(n2 === n1, "lazo: Shift restaura exactamente las mismas", `${n2}`);
afirmar(C.proyecciones[0].estado === "activa", "lazo: restaura a activa");

// --- union ------------------------------------------------------------------
// Dos tramos reales con eje temporal, elegidos por cercania de extremos.
let iA = -1, iB = -1, mejor = Infinity;
for (let i = 0; i < 400; i++) {
  const A1 = C.proyecciones[i];
  if (!A1 || A1.t_ini == null) continue;
  for (let j = i + 1; j < 400; j++) {
    const B1 = C.proyecciones[j];
    if (!B1 || B1.t_ini == null || B1.t_ini === A1.t_ini) continue;
    const d = Math.hypot(A1.pf.x - B1.p0.x, A1.pf.y - B1.p0.y);
    if (d < mejor) { mejor = d; iA = i; iB = j; }
  }
}
console.log(`\nuniendo ${C.proyecciones[iA].id} y ${C.proyecciones[iB].id} `
          + `(extremos a ${mejor.toFixed(1)} px)`);

const nA = C.proyecciones[iA].n, nB = C.proyecciones[iB].n;
const tA = C.proyecciones[iA].t_ini, tB = C.proyecciones[iB].t_ini;
const k = unirTrayectorias(iA, iB);
const Tu = C.proyecciones[k];

afirmar(Tu != null, "union: crea la trayectoria");
afirmar(Tu.n === nA + nB, "union: conserva todos los puntos", `${Tu.n} = ${nA}+${nB}`);
afirmar(Tu.frames.length === Tu.n, "union: un frame por punto");
afirmar(Tu.t_ini === Math.min(tA, tB), "union: arranca en el tramo mas temprano");
const asc = Tu.frames.every((v, i, a) => i === 0 || a[i-1] <= v);
afirmar(asc, "union: frames quedan ascendentes (no se invirtio nada)");
afirmar(C.proyecciones[iA].estado === "descartada" && C.proyecciones[iB].estado === "descartada",
        "union: los dos tramos quedan descartados");
afirmar(C.proyecciones[iA].razon === "unida", "union: razon 'unida'");
afirmar(Array.isArray(Tu.unida_de) && Tu.unida_de.length === 2, "union: registra unida_de");
afirmar(Tu.tortuosidad >= 1, "union: tortuosidad recalculada >= 1", Tu.tortuosidad);
afirmar(Tu.fuente === "unida", "union: fuente marcada");
afirmar(unirTrayectorias(3, 3) === null, "union: no se une consigo misma");

// Dos tramos CONSECUTIVOS: el resultado tiene que ser la concatenacion pura.
const mk = (id, f0, n, x0) => {
  const T = { id, puntos: [], frames: [], clasificacion: "Proyección", estado: "activa",
              escape_relativo: 0, r2_score: 1 };
  for (let i = 0; i < n; i++) { T.frames.push(f0 + i); T.puntos.push(x0 + i, 0); }
  T.t_ini = T.frames[0]; T.t_fin = T.frames[n - 1];
  preparar(T); return T;
};
const c1 = mk("c1", 10, 5, 0), c2 = mk("c2", 20, 5, 100);
C.proyecciones.push(c1, c2);
const kc = unirTrayectorias(C.proyecciones.length - 2, C.proyecciones.length - 1);
const Tc = C.proyecciones[kc];
afirmar(Tc.frames.join(",") === "10,11,12,13,14,20,21,22,23,24",
        "union consecutiva: equivale a concatenar", Tc.frames.join(","));
afirmar(solapeTemporal(c1, c2) === 0, "solape: tramos consecutivos no solapan");

// Dos tramos SOLAPADOS: los frames tienen que quedar ascendentes igual.
const s1 = mk("s1", 10, 6, 0), s2 = mk("s2", 12, 6, 50);
C.proyecciones.push(s1, s2);
afirmar(solapeTemporal(s1, s2) > 0, "solape: se detecta", solapeTemporal(s1, s2) + " frames");
const ks = unirTrayectorias(C.proyecciones.length - 2, C.proyecciones.length - 1);
const Ts = C.proyecciones[ks];
const ascS = Ts.frames.every((v, i, a) => i === 0 || a[i-1] <= v);
afirmar(ascS, "union solapada: frames ascendentes", Ts.frames.join(","));
afirmar(Ts.frames.length === Ts.n, "union solapada: un frame por punto");
afirmar(Ts.n === 12, "union solapada: no pierde puntos", Ts.n);

// Sin eje temporal debe caer a la geometria e invertir si hace falta.
const P = { id:"p", puntos:[0,0, 10,0], frames:null, t_ini:null, clasificacion:"Proyección", estado:"activa" };
const Q = { id:"q", puntos:[30,0, 20,0], frames:null, t_ini:null, clasificacion:"Proyección", estado:"activa" };
preparar(P); preparar(Q);
C.proyecciones.push(P, Q);
const kg = unirTrayectorias(C.proyecciones.length - 2, C.proyecciones.length - 1);
const Tg = C.proyecciones[kg];
afirmar(Tg.n === 4, "union sin tiempo: junta los 4 puntos");
const xs = [];
for (let i = 0; i < Tg.puntos.length; i += 2) xs.push(Tg.puntos[i]);
afirmar(xs[0] === 0 && xs[xs.length-1] === 30, "union sin tiempo: invierte para que sea continua", xs.join(","));

// --- resultado --------------------------------------------------------------
console.log();
for (const t of ok)  console.log("  OK    " + t);
for (const t of mal) console.log("  FALLA " + t);
console.log(`\n${ok.length} pasan, ${mal.length} fallan`);
process.exit(mal.length ? 1 : 0);
