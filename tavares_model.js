/*
 * tavares_model.js — port fiel (vanilla JS, sem dependências) do núcleo
 * tavares_sfc.py. Mesmas fórmulas, projeção no domínio físico, softmax SIGNED
 * e contagem de passos. Usado pelo dashboard (globalThis.TavaresModel) e pelo
 * teste de fidelidade node (module.exports).
 *
 * Estado 7D: x = (F, L, M, D, u, s, rho). Índices algébricos Phi/Omega/Sigma.
 * A trajetória (traj/probs/pressure) é DETERMINÍSTICA: rhs usa stationary_probs,
 * não o estado amostrado de Markov. Só `regimes` depende do RNG (mulberry32),
 * que difere do PCG64 do numpy — por isso regimes NÃO é comparado bit a bit.
 */
(function (root) {
  "use strict";

  // ----------------------------------------------------------------------- //
  // Parâmetros (calibração ILUSTRATIVA — não são dados observados)
  // ----------------------------------------------------------------------- //
  function defaultParams() {
    return {
      delta: 0.12, iota: 0.50, eps: 1.30, r_bar: 0.03, zeta: 2.0,
      r_L: 0.020, theta_ind: 0.70, alpha: 0.60, beta: 0.50, gamma: 1.00,
      r_f0: 0.030, kappa: 0.020, nu_info: 1.00, mu: 0.25, chi: 0.80, lam: 0.50,
      a_M: 0.40, eta_M: 0.10, m_u: 0.30,
      tau_w: 0.30, tau_c: 0.25, w0: 0.55, phi0: 0.15, phi1: 0.30, tau_escape: 0.40,
      g0: 0.06, g1: 0.18, G_infra: 0.05, sp0: 0.010, sp_s: 0.050, sp_rho: 0.030,
      a_u: 0.15, b_u: 0.10, c_u: 0.05,
      a_s: 0.020, b_s: 0.50, d_s: 0.05, p_rho: 0.80, q_rho: 0.60, r0_rho: 0.10, c_rho: 0.05,
      L_half: 0.10, fiscal_resp: 1.00,
      Phi_star: 1.00, Omega_star: 1.50, Sigma_star: 0.50,
      tau_temp: 0.30, eps_mix: 0.15, w1Phi: 1.0, w1Sigma: 1.0, w2Sigma: 1.2, w3Omega: 1.0, w4: 1.3,
      F_bar: 1.0,
    };
  }

  const REGIMES = {
    1: "Centralização oligopolista por queima de capital",
    2: "Crise sistêmica monetária e soberana",
    3: "Barbárie social e paralisia de governabilidade",
    4: "Reconfiguração estatal top-down",
  };

  // ----------------------------------------------------------------------- //
  // Auxiliares
  // ----------------------------------------------------------------------- //
  const softplus = (z) => (z < 30 ? Math.log1p(Math.exp(z)) : z);
  const sigmoid = (z) => (z >= -30 ? 1.0 / (1.0 + Math.exp(-z)) : 0.0);
  const vadd = (a, b) => a.map((v, i) => v + b[i]);
  const vscale = (a, s) => a.map((v) => v * s);
  function argmax(a) { let k = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[k]) k = i; return k; }
  function linspace(a, b, n) {
    if (n <= 1) return [a];
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = a + (b - a) * i / (n - 1);
    return out;
  }

  function investment(F, L, M, p) {
    return p.iota * Math.pow(Math.max(M, 0.0), p.eps) / (1.0 + Math.pow(p.r_L / p.r_bar, p.zeta));
  }
  function profits(F, L, M, p) {
    const pi_NO = (p.r_f0 + p.kappa * p.nu_info) * Math.max(L, 0.0);
    const pi_OP = p.mu * Math.max(M, 0.0);
    return [pi_NO, pi_OP];
  }
  function sovereignRate(s, rho, p) {
    return p.r_L + p.sp0 + p.sp_s * Math.max(s, 0.0) + p.sp_rho * (1.0 - rho);
  }
  function wageBill(u, M, p) {
    const Y = 1.0 + 0.5 * Math.max(M, 0.0);
    const W = p.w0 * (1.0 - u) * Y;
    const phi = Math.min(0.95, p.phi0 + p.phi1 * u);
    return [phi * W, (1.0 - phi) * W, W];
  }

  function indices(x, p) {
    const [F, L, M, D, u, s, rho] = x;
    const pi_OP = profits(F, L, M, p)[1];
    const FCL_IA = p.chi * pi_OP;
    const Phi = p.r_L * p.theta_ind * Math.max(F, 0.0) / (FCL_IA + p.lam * Math.max(L, 0.0) + 1e-9);
    const [W_p, W_o, W] = wageBill(u, M, p);
    const Pi_local = (1.0 - p.tau_escape) * pi_OP;
    const T = p.tau_w * W + p.tau_c * Pi_local;
    const i_sov = sovereignRate(s, rho, p);
    const G_soc = p.g0 + p.g1 * u;
    const Omega = (G_soc + p.G_infra + i_sov * Math.max(D, 0.0)) / (T + 1e-9);
    const Sigma = p.nu_info * Math.max(s, 0.0) * (1.0 - rho);
    return { Phi, Omega, Sigma, FCL_IA, T, i_sov, G_soc, W_p, W_o, Pi_local };
  }

  function netWorthGrowth(x, p) {
    const [F, L, M] = x;
    const [pi_NO, pi_OP] = profits(F, L, M, p);
    const I = investment(F, L, M, p);
    const dF = I - p.delta * F;
    const dL = p.alpha * pi_NO + p.beta * pi_OP - I - p.gamma * p.r_L * F;
    return dF + dL;
  }

  // ----------------------------------------------------------------------- //
  // Markov endógeno: potenciais SIGNED -> softmax
  // ----------------------------------------------------------------------- //
  function potentialsFromIndices(Phi, Omega, Sigma, dK, p) {
    const zPhi = Phi - p.Phi_star;
    const zOmega = Omega - p.Omega_star;
    const zSigma = Sigma - p.Sigma_star;
    const V1 = p.w1Phi * zPhi - p.w1Sigma * zSigma;
    const V2 = p.w2Sigma * zSigma;
    const V3 = p.w3Omega * zOmega * sigmoid(5.0 * dK);
    const V4 = p.w4 * Math.min(zOmega, zPhi);
    return [V1, V2, V3, V4];
  }
  function regimePotentials(x, p) {
    const idx = indices(x, p);
    return potentialsFromIndices(idx.Phi, idx.Omega, idx.Sigma, netWorthGrowth(x, p), p);
  }
  function stationaryProbs(x, p) {
    let V = regimePotentials(x, p).map((v) => v / p.tau_temp);
    const m = Math.max(...V);
    V = V.map((v) => v - m);
    const e = V.map((v) => Math.exp(v));
    const sum = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / sum);
  }
  function transitionMatrix(x, p) {
    const pi = stationaryProbs(x, p);
    const P = [];
    for (let i = 0; i < 4; i++) {
      const row = [];
      let s = 0;
      for (let j = 0; j < 4; j++) {
        const P0 = i === j ? 0.70 : 0.10 / 3.0;
        const v = (1.0 - p.eps_mix) * pi[j] + p.eps_mix * P0;
        row.push(v); s += v;
      }
      P.push(row.map((v) => v / s));
    }
    return P;
  }

  // ----------------------------------------------------------------------- //
  // Domínio físico e RHS
  // ----------------------------------------------------------------------- //
  const _LOWER = [0, 0, 0, 0, 0, 0, 0];
  const _UPPER = [Infinity, Infinity, 1, Infinity, 1, Infinity, 1];
  function project(x) {
    return x.map((v, i) => Math.min(_UPPER[i], Math.max(_LOWER[i], v)));
  }

  function rhs(t, x, p, markovFeedback) {
    if (markovFeedback === undefined) markovFeedback = true;
    const [F, L, M, D, u, s, rho] = project(x);
    const [pi_NO, pi_OP] = profits(F, L, M, p);
    const I_raw = investment(F, L, M, p);
    const I = I_raw * L / (L + p.L_half);
    const idx = indices([F, L, M, D, u, s, rho], p);
    const i_sov = idx.i_sov, T = idx.T;
    const probs = markovFeedback ? stationaryProbs([F, L, M, D, u, s, rho], p) : [0.25, 0.25, 0.25, 0.25];
    const P2 = probs[1], P4 = probs[3];
    const T_eff = T * (1.0 + p.fiscal_resp * P4);

    const dF = I - p.delta * F;
    const dL = p.alpha * pi_NO + p.beta * pi_OP - I - p.gamma * p.r_L * F;
    const dM = p.a_M * M * (1.0 - M) * (1.0 + p.eta_M * p.nu_info) - p.m_u * u * M;
    const dD = idx.G_soc + p.G_infra + i_sov * D - T_eff;
    const du = p.a_u * (F / p.F_bar) * (1.0 - u) - p.b_u * M * (1.0 - u) - p.c_u * u;
    const stress = softplus(idx.Sigma - p.Sigma_star);
    const ds = p.a_s * (1.0 - rho) + p.d_s * stress - p.b_s * rho * s;
    const drho = (1.0 - rho) * (p.r0_rho + p.q_rho * P2) - rho * (p.p_rho * s + p.c_rho);
    return [dF, dL, dM, dD, du, ds, drho];
  }

  function rk4Step(t, x, p, dt, markovFeedback) {
    const k1 = rhs(t, x, p, markovFeedback);
    const k2 = rhs(t + dt / 2, vadd(x, vscale(k1, dt / 2)), p, markovFeedback);
    const k3 = rhs(t + dt / 2, vadd(x, vscale(k2, dt / 2)), p, markovFeedback);
    const k4 = rhs(t + dt, vadd(x, vscale(k3, dt)), p, markovFeedback);
    const comb = k1.map((v, i) => v + 2 * k2[i] + 2 * k3[i] + k4[i]);
    return project(vadd(x, vscale(comb, dt / 6)));
  }

  function integrate(x0, p, tEnd, nSteps, markovFeedback) {
    if (tEnd === undefined) tEnd = 40.0;
    if (nSteps === undefined) nSteps = 8000;
    if (markovFeedback === undefined) markovFeedback = true;
    let x = project(x0.slice());
    const h = tEnd / (nSteps - 1);
    const xs = [x.slice()];
    for (let k = 1; k < nSteps; k++) {
      x = rk4Step((k - 1) * h, x, p, h, markovFeedback);
      xs.push(x.slice());
    }
    return { t: linspace(0.0, tEnd, nSteps), xs };
  }

  // ----------------------------------------------------------------------- //
  // Termômetro Composto de Transição de Regime
  // ----------------------------------------------------------------------- //
  function thermometerFromIndices(Phi, Omega, Sigma, dK, p) {
    let V = potentialsFromIndices(Phi, Omega, Sigma, dK, p).map((v) => v / p.tau_temp);
    const mx = Math.max(...V);
    V = V.map((v) => v - mx);
    const e = V.map((v) => Math.exp(v));
    const sum = e.reduce((a, b) => a + b, 0);
    const pi = e.map((v) => v / sum);
    const m = {
      dPhi: Phi - p.Phi_star, dOmega: Omega - p.Omega_star,
      dSigma: Sigma - p.Sigma_star, debt_sustainability: Omega - 1.0,
    };
    const zPhi = Math.max(0.0, m.dPhi), zOmega = Math.max(0.0, m.dOmega), zSigma = Math.max(0.0, m.dSigma);
    const pressure = 1.0 - Math.exp(-(zPhi * zPhi + zOmega * zOmega + zSigma * zSigma));
    const dominant = argmax(pi) + 1;
    const regime_probs = {};
    for (let i = 0; i < 4; i++) regime_probs[REGIMES[i + 1]] = pi[i];
    return {
      indices: { Phi, Omega, Sigma },
      regime_probs,
      dominant_regime: dominant,
      dominant_label: REGIMES[dominant],
      transition_pressure: pressure,
      margins: m,
      net_worth_growth: dK,
    };
  }
  function thermometer(x, p) {
    const idx = indices(x, p);
    const th = thermometerFromIndices(idx.Phi, idx.Omega, idx.Sigma, netWorthGrowth(x, p), p);
    th.indices = idx;
    return th;
  }

  // ----------------------------------------------------------------------- //
  // Simulação acoplada ODE + Markov (RNG semeado p/ reprodutibilidade)
  // ----------------------------------------------------------------------- //
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function choice(rng, probs) {
    const r = rng();
    let cum = 0;
    for (let i = 0; i < probs.length; i++) { cum += probs[i]; if (r < cum) return i; }
    return probs.length - 1;
  }

  function simulateCoupled(x0, p, tEnd, dt, seed, markovFeedback) {
    if (tEnd === undefined) tEnd = 40.0;
    if (dt === undefined) dt = 0.05;
    if (seed === undefined) seed = 42;
    if (markovFeedback === undefined) markovFeedback = true;
    const rng = mulberry32(seed);
    let x = project(x0.slice());
    const n = Math.trunc(tEnd / dt);
    const traj = [x.slice()];
    const regimes = [];
    const probs = [];
    const pressure = [];
    let state = argmax(stationaryProbs(x, p));
    regimes.push(state + 1);
    probs.push(stationaryProbs(x, p));
    pressure.push(thermometer(x, p).transition_pressure);
    for (let k = 1; k <= n; k++) {
      x = rk4Step((k - 1) * dt, x, p, dt, markovFeedback);
      const P = transitionMatrix(x, p);
      state = choice(rng, P[state]);
      traj.push(x.slice());
      regimes.push(state + 1);
      probs.push(stationaryProbs(x, p));
      pressure.push(thermometer(x, p).transition_pressure);
    }
    return { t: linspace(0, tEnd, n + 1), traj, regimes, probs, pressure };
  }

  const api = {
    defaultParams, REGIMES,
    softplus, sigmoid, investment, profits, sovereignRate, wageBill,
    indices, netWorthGrowth,
    potentialsFromIndices, regimePotentials, stationaryProbs, transitionMatrix,
    project, rhs, rk4Step, integrate,
    thermometerFromIndices, thermometer,
    mulberry32, simulateCoupled,
    linspace, argmax,
  };

  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.TavaresModel = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
