/*
 * test_model.js — portão de fidelidade do port JS (tavares_model.js) contra o
 * núcleo Python (tavares_sfc.py), via referências gravadas em dashboard_data.js.
 *
 * Compara APENAS a parte determinística: índices algébricos, termômetro no ponto
 * fixo, endpoint de integrate e a trajetória/probs/pressão de simulateCoupled.
 * `regimes` NÃO é comparado (depende do RNG: mulberry32 != PCG64 do numpy).
 *
 * Uso: node test_model.js   (requer dashboard_data.js gerado por --export)
 */
"use strict";
const fs = require("fs");
const path = require("path");

const dataSrc = fs.readFileSync(path.join(__dirname, "dashboard_data.js"), "utf8");
const DATA = new Function(dataSrc + "\n; return DATA;")();
const M = require("./tavares_model.js");

const ref = DATA.ref;
const p = Object.assign(M.defaultParams(), ref.params);
const x0 = ref.x0;
const fp = ref.fp_x;

const checks = [];
function check(name, ok, detail) { checks.push({ name, ok, detail }); }
function maxDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}
function maxDiffRows(A, B) {
  let m = 0;
  for (let i = 0; i < A.length; i++) m = Math.max(m, maxDiff(A[i], B[i]));
  return m;
}
const probsArray = (th) => [1, 2, 3, 4].map((i) => th.regime_probs[M.REGIMES[i]]);

// 0) defaults batem com os parâmetros de referência
{
  const keys = Object.keys(ref.params);
  let bad = keys.filter((k) => Math.abs(M.defaultParams()[k] - ref.params[k]) > 1e-12);
  check("params defaults == ref.params", bad.length === 0, bad.length ? "diff: " + bad.join(",") : keys.length + " campos");
}

// 1) índices algébricos no estado inicial
{
  const js = M.indices(x0, p);
  const keys = Object.keys(ref.indices_x0);
  let m = 0; for (const k of keys) m = Math.max(m, Math.abs(js[k] - ref.indices_x0[k]));
  check("indices(x0) == ref.indices_x0", m < 1e-9, "max|Δ|=" + m.toExponential(3));
}

// 2) índices algébricos no ponto fixo
{
  const js = M.indices(fp, p);
  const keys = Object.keys(ref.fp_indices);
  let m = 0; for (const k of keys) m = Math.max(m, Math.abs(js[k] - ref.fp_indices[k]));
  check("indices(fp) == ref.fp_indices", m < 1e-9, "max|Δ|=" + m.toExponential(3));
}

// 3) termômetro no ponto fixo (pressão, dominante, probs)
{
  const th = M.thermometer(fp, p);
  const dp = Math.abs(th.transition_pressure - ref.fp_pressure);
  const domOk = th.dominant_regime === ref.fp_dominant;
  const dpr = maxDiff(probsArray(th), ref.fp_probs);
  check("thermometer(fp) pressão/dominante/probs", dp < 1e-9 && domOk && dpr < 1e-9,
    `pressão Δ=${dp.toExponential(2)} dom=${th.dominant_regime}/${ref.fp_dominant} probs Δ=${dpr.toExponential(2)}`);
}

// 4) integrate endpoint (RK4 8000 passos, h=t_end/(n-1))
{
  const { t, xs } = M.integrate(x0, p, 40.0, 8000);
  const lenOk = xs.length === 8000 && t.length === 8000;
  const m = maxDiff(xs[xs.length - 1], ref.integrate_end);
  check("integrate endpoint == ref.integrate_end", lenOk && m < 1e-3,
    `len=${xs.length} max|Δ|=${m.toExponential(3)}`);
}

// 5) simulateCoupled: trajetória/probs/pressão determinísticas (todos os pontos)
{
  const sim = M.simulateCoupled(x0, p, 40.0, 0.05, 42);
  const base = DATA.model.baseline_sim;
  const lenOk = sim.traj.length === base.traj.length;
  const dTraj = maxDiffRows(sim.traj, base.traj);
  const dProbs = maxDiffRows(sim.probs, base.probs);
  const dPress = maxDiff(sim.pressure, base.pressure);
  check("simulateCoupled traj == baseline (todos os pontos)", lenOk && dTraj < 1e-3,
    `len=${sim.traj.length}/${base.traj.length} max|Δ|=${dTraj.toExponential(3)}`);
  check("simulateCoupled probs == baseline", dProbs < 1e-3, "max|Δ|=" + dProbs.toExponential(3));
  check("simulateCoupled pressure == baseline", dPress < 1e-3, "max|Δ|=" + dPress.toExponential(3));
  const dx = maxDiff(sim.traj[sim.traj.length - 1], ref.sim_end_x);
  const dpe = maxDiff(sim.probs[sim.probs.length - 1], ref.sim_end_probs);
  const dpr = Math.abs(sim.pressure[sim.pressure.length - 1] - ref.sim_end_pressure);
  check("simulateCoupled endpoint == ref.sim_end_*", dx < 1e-3 && dpe < 1e-3 && dpr < 1e-3,
    `x Δ=${dx.toExponential(2)} probs Δ=${dpe.toExponential(2)} pressão Δ=${dpr.toExponential(2)}`);
  // regimes é estocástico: só sanity (1..4, mesmo comprimento), NÃO igualdade
  const regOk = sim.regimes.length === sim.traj.length && sim.regimes.every((r) => r >= 1 && r <= 4);
  check("regimes sanity (estocástico, não comparado)", regOk, "len=" + sim.regimes.length);
}

let fail = 0;
console.log("=".repeat(72));
console.log("FIDELIDADE JS <-> PYTHON  (tavares_model.js vs dashboard_data.js ref)");
console.log("=".repeat(72));
for (const c of checks) {
  console.log(`${c.ok ? "PASS" : "FAIL"}  ${c.name}  [${c.detail}]`);
  if (!c.ok) fail++;
}
console.log("-".repeat(72));
console.log(fail === 0 ? `TODOS OS ${checks.length} TESTES PASSARAM` : `${fail}/${checks.length} TESTES FALHARAM`);
process.exit(fail === 0 ? 0 : 1);
