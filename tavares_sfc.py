"""
Modelo SFC-ODE-Markov da economia política de Maria da Conceição Tavares (1992).

Traduz financeirização, concorrência schumpeteriana, descarte do trabalho vivo e
crise do Estado num sistema dinâmico não-linear com consistência estoque-fluxo
(SFC) e matriz de transição de Markov endógena. Rejeita equilíbrio geral: o
capitalismo opera em desequilíbrio contínuo (retornos crescentes + assimetria
monetária hierárquica).

Núcleo depende apenas de numpy. scipy (integração/raízes) e matplotlib (retratos
de fase) são opcionais e detectados em tempo de import.

Vetor de estado (7D), tudo normalizado:
    x = (F, L, M, D, u, s, rho)
    F   : capital fixo tecnológico ilíquido e indivisível (data centers, chips, energia)
    L   : reserva líquida corporativa (tesouraria das Big Techs)
    M   : alcance operacional de mercado (base instalada / receita), em [0,1]
    D   : dívida soberana do polo emissor (EUA)
    u   : desemprego tecnológico (descarte do trabalho vivo)
    s   : variância de divergência geoeconômica sigma^2_{FX,r}
    rho : coordenação entre bancos centrais / políticas, em [0,1]

Índices de tensão (algébricos, não-estados):
    Phi = r_L * theta_ind * F / (FCL_IA + lambda * L)   ponte dinheiro-máquinas
    Omega = (G_soc(u) + G_infra + i_sov*D) / T          sobrecarga fiscal/previdenciária
    Sigma = nu_info * s * (1 - rho)                     risco sistêmico / efeito manada

Correção SFC em relação à especificação original: o prompt escreve
    dK = alpha*pi_NO + beta*pi_OP - gamma*r_L*I_ind
cobrando carregamento sobre o FLUXO de investimento e ignorando depreciação.
Aqui K = F + L e o bloco corporativo é separado em acumulação física (dF) e
acumulação de liquidez (dL); a regra de valorização do prompt emerge como forma
reduzida dK = alpha*pi_NO + beta*pi_OP - gamma*r_L*F - delta*F (carregamento sobre
o ESTOQUE ilíquido + depreciação moral schumpeteriana).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np

try:  # opcionais
    from scipy.optimize import root
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # pragma: no cover
    _HAVE_MPL = False


# --------------------------------------------------------------------------- #
# Parâmetros
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    # bloco corporativo (tripé de acumulação)
    delta: float = 0.12      # depreciação moral do capital de IA (obsolescência rápida)
    iota: float = 0.50       # escala do investimento indivisível
    eps: float = 1.30        # expoente de retornos crescentes de escala (>1)
    r_bar: float = 0.03      # escala de estrangulamento pela taxa longa
    zeta: float = 2.0        # sensibilidade do investimento à taxa longa
    r_L: float = 0.020       # taxa de juros real longa (driver exógeno)
    theta_ind: float = 0.70  # indivisibilidade: fração do capital em ativos físicos pesados
    alpha: float = 0.60      # retenção/valorização do lucro financeiro
    beta: float = 0.50       # retenção/valorização do lucro operacional
    gamma: float = 1.00      # peso do custo de carregamento do estoque ilíquido
    r_f0: float = 0.030      # rendimento financeiro base da tesouraria
    kappa: float = 0.020     # ganho informacional/algorítmico sobre o rendimento
    nu_info: float = 1.00    # velocidade informacional/algorítmica (normalizada)
    mu: float = 0.25         # margem operacional (compressível taticamente)
    chi: float = 0.80        # conversão do lucro operacional em FCL_IA
    lam: float = 0.50        # usabilidade do colchão líquido (lambda)

    # mercado
    a_M: float = 0.40        # taxa logística de expansão de mercado
    eta_M: float = 0.10      # boost algorítmico à conquista de mercado
    m_u: float = 0.30        # erosão de demanda pelo desemprego tecnológico

    # trabalho e Estado
    tau_w: float = 0.30      # alíquota sobre a massa salarial
    tau_c: float = 0.25      # alíquota sobre o lucro local tributável
    w0: float = 0.55         # participação salarial baseline no PIB
    phi0: float = 0.15       # fração da elite de produtores de tecnologia em W
    phi1: float = 0.30       # aumento da fração da elite com a automação
    tau_escape: float = 0.40 # desterritorialização: lucro que escapa à base local
    g0: float = 0.06         # gasto social autônomo
    g1: float = 0.18         # gasto social induzido pelo desemprego
    G_infra: float = 0.05    # gasto público em infraestrutura
    sp0: float = 0.010       # spread soberano base
    sp_s: float = 0.050      # spread sensível à divergência
    sp_rho: float = 0.030    # spread sensível à falta de coordenação

    # dinâmica laboral
    a_u: float = 0.15        # deslocamento por automação (trabalho morto)
    b_u: float = 0.10        # reabsorção via novas tarefas operacionais
    c_u: float = 0.05        # rotatividade natural

    # bloco monetário/geoeconômico
    a_s: float = 0.020       # forcing de divergência sob não-coordenação
    b_s: float = 0.50        # amortecimento da divergência pela coordenação
    d_s: float = 0.05        # amplificação da divergência pelo estresse sistêmico
    p_rho: float = 0.80      # erosão da coordenação pela divergência
    q_rho: float = 0.60      # resposta emergencial de coordenação na crise (Fed LOLR)
    r0_rho: float = 0.10     # cooperação baseline (piso institucional da coordenação)
    c_rho: float = 0.05      # decaimento natural da coordenação

    # restrições de financiamento e reação fiscal endógena
    L_half: float = 0.10     # gating de liquidez do investimento (Michaelis-Menten)
    fiscal_resp: float = 1.00  # resposta de receita do Regime 4 (tributação/reforma)

    # limiares de bifurcação / atractores
    Phi_star: float = 1.00   # tensão crítica da ponte dinheiro-máquinas
    Omega_star: float = 1.50 # tolerância política/fiscal crítica
    Sigma_star: float = 0.50 # risco sistêmico crítico

    # Markov endógeno
    tau_temp: float = 0.30   # temperatura do softmax de atractores
    eps_mix: float = 0.15    # mistura com persistência baseline
    w1Phi: float = 1.0
    w1Sigma: float = 1.0
    w2Sigma: float = 1.2
    w3Omega: float = 1.0
    w4: float = 1.3

    F_bar: float = 1.0       # normalização da intensidade de capital de IA

    @classmethod
    def illustrative(cls) -> "Params":
        """Calibração ILUSTRATIVA (não são dados observados). Ver PROXIES para
        alimentar o modelo com séries reais via load_fred()/seu próprio loader."""
        return cls()


# --------------------------------------------------------------------------- #
# Funções auxiliares
# --------------------------------------------------------------------------- #
def _softplus(z: float) -> float:
    return math.log1p(math.exp(z)) if z < 30 else z


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z)) if z >= -30 else 0.0


def investment(F, L, M, p: Params) -> float:
    """Investimento indivisível I_ind: retornos crescentes em M, estrangulado por r_L."""
    return p.iota * max(M, 0.0) ** p.eps / (1.0 + (p.r_L / p.r_bar) ** p.zeta)


def profits(F, L, M, p: Params):
    pi_NO = (p.r_f0 + p.kappa * p.nu_info) * max(L, 0.0)   # lucro financeiro não-operacional
    pi_OP = p.mu * max(M, 0.0)                              # lucro operacional
    return pi_NO, pi_OP


def sovereign_rate(s, rho, p: Params) -> float:
    return p.r_L + p.sp0 + p.sp_s * max(s, 0.0) + p.sp_rho * (1.0 - rho)


def wage_bill(u, M, p: Params):
    """Bifurcação laboral: elite W_p vs massa precarizada W_o (putting-out digital)."""
    Y = 1.0 + 0.5 * max(M, 0.0)            # renda agregada normalizada
    W = p.w0 * (1.0 - u) * Y               # massa salarial total
    phi = min(0.95, p.phi0 + p.phi1 * u)   # fração da elite cresce com a automação
    W_p, W_o = phi * W, (1.0 - phi) * W
    return W_p, W_o, W


def indices(x, p: Params) -> dict:
    F, L, M, D, u, s, rho = x
    _, pi_OP = profits(F, L, M, p)
    FCL_IA = p.chi * pi_OP                          # fluxo de caixa livre da IA
    Phi = p.r_L * p.theta_ind * max(F, 0.0) / (FCL_IA + p.lam * max(L, 0.0) + 1e-9)
    W_p, W_o, W = wage_bill(u, M, p)
    Pi_local = (1.0 - p.tau_escape) * pi_OP         # base tributável local
    T = p.tau_w * W + p.tau_c * Pi_local            # arrecadação
    i_sov = sovereign_rate(s, rho, p)
    G_soc = p.g0 + p.g1 * u
    Omega = (G_soc + p.G_infra + i_sov * max(D, 0.0)) / (T + 1e-9)
    Sigma = p.nu_info * max(s, 0.0) * (1.0 - rho)
    return dict(Phi=Phi, Omega=Omega, Sigma=Sigma, FCL_IA=FCL_IA, T=T,
                i_sov=i_sov, G_soc=G_soc, W_p=W_p, W_o=W_o, Pi_local=Pi_local)


def net_worth_growth(x, p: Params) -> float:
    """dK = dF + dL (regra de valorização SFC-consistente do prompt)."""
    F, L, M, D, u, s, rho = x
    pi_NO, pi_OP = profits(F, L, M, p)
    I = investment(F, L, M, p)
    dF = I - p.delta * F
    dL = p.alpha * pi_NO + p.beta * pi_OP - I - p.gamma * p.r_L * F
    return dF + dL


# --------------------------------------------------------------------------- #
# Markov endógeno: atractores como potenciais de um softmax sobre os limiares
# --------------------------------------------------------------------------- #
REGIMES = {
    1: "Centralização oligopolista por queima de capital",
    2: "Crise sistêmica monetária e soberana",
    3: "Barbárie social e paralisia de governabilidade",
    4: "Reconfiguração estatal top-down",
}


def potentials_from_indices(Phi, Omega, Sigma, dK, p: Params) -> np.ndarray:
    """Núcleo dos potenciais SIGNED dado (Phi,Omega,Sigma,dK). Fonte única de
    verdade: usada tanto pelo estado do modelo quanto por índices empíricos reais
    (tavares_data), garantindo que o mesmo limiar se aplique à mesma grandeza."""
    zPhi = Phi - p.Phi_star
    zOmega = Omega - p.Omega_star
    zSigma = Sigma - p.Sigma_star
    V1 = p.w1Phi * zPhi - p.w1Sigma * zSigma          # queima de capital, sem crise sistêmica
    V2 = p.w2Sigma * zSigma                            # explosão monetária/soberana
    V3 = p.w3Omega * zOmega * _sigmoid(5.0 * dK)       # sucesso privado + colapso social
    V4 = p.w4 * min(zOmega, zPhi)                      # crise dupla -> intervenção estatal
    return np.array([V1, V2, V3, V4])


def regime_potentials(x, p: Params) -> np.ndarray:
    """Potenciais SIGNED dos atractores. Abaixo dos limiares todos são negativos:
    o softmax devolve a atração RELATIVA (regime dominante = limiar mais próximo).
    A pressão composta (ver thermometer) é que indica se alguma transição está viva."""
    idx = indices(x, p)
    return potentials_from_indices(idx["Phi"], idx["Omega"], idx["Sigma"],
                                   net_worth_growth(x, p), p)


def stationary_probs(x, p: Params) -> np.ndarray:
    V = regime_potentials(x, p)
    V = V / p.tau_temp
    V = V - V.max()
    e = np.exp(V)
    return e / e.sum()


def transition_matrix(x, p: Params) -> np.ndarray:
    """P_ij endógena: mistura a distribuição de atractores com persistência baseline."""
    pi = stationary_probs(x, p)
    P0 = np.full((4, 4), 0.10 / 3.0)
    np.fill_diagonal(P0, 0.70)
    P = (1.0 - p.eps_mix) * np.tile(pi, (4, 1)) + p.eps_mix * P0
    return P / P.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Sistema de equações diferenciais (SFC)
# --------------------------------------------------------------------------- #
# domínio físico: F,L,D,s >= 0 ; M,u,rho em [0,1]
_LOWER = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_UPPER = np.array([np.inf, np.inf, 1.0, np.inf, 1.0, np.inf, 1.0])


def project(x):
    """Projeção no domínio físico. Os pisos representam restrições institucionais:
    L=0 -> congelamento de financiamento; rho=0 -> não-coordenação total."""
    return np.clip(x, _LOWER, _UPPER)


def rhs(t, x, p: Params, markov_feedback: bool = True):
    F, L, M, D, u, s, rho = project(np.asarray(x, dtype=float))

    pi_NO, pi_OP = profits(F, L, M, p)
    # investimento indivisível gated por liquidez: sem tesouraria, a acumulação trava
    I_raw = investment(F, L, M, p)
    I = I_raw * L / (L + p.L_half)
    idx = indices((F, L, M, D, u, s, rho), p)
    i_sov, T = idx["i_sov"], idx["T"]
    probs = stationary_probs((F, L, M, D, u, s, rho), p) if markov_feedback \
        else np.full(4, 0.25)
    P2, P4 = probs[1], probs[3]
    # reação fiscal endógena do Regime 4: tributação/reforma eleva a receita efetiva
    T_eff = T * (1.0 + p.fiscal_resp * P4)

    # (1) acumulação física do capital indivisível
    dF = I - p.delta * F
    # (2) acumulação de liquidez (lucros retidos - investimento - carregamento do estoque)
    dL = p.alpha * pi_NO + p.beta * pi_OP - I - p.gamma * p.r_L * F
    # (3) conquista de mercado: logística com retornos crescentes + erosão laboral
    dM = p.a_M * M * (1.0 - M) * (1.0 + p.eta_M * p.nu_info) - p.m_u * u * M
    # (4) dívida soberana: identidade SFC dD = G + i_sov*D - T_eff
    dD = idx["G_soc"] + p.G_infra + i_sov * D - T_eff
    # (5) descarte do trabalho vivo (automação > reabsorção)
    du = p.a_u * (F / p.F_bar) * (1.0 - u) - p.b_u * M * (1.0 - u) - p.c_u * u
    # (6) divergência geoeconômica (limitada superiormente pela coordenação)
    stress = _softplus(idx["Sigma"] - p.Sigma_star)
    ds = p.a_s * (1.0 - rho) + p.d_s * stress - p.b_s * rho * s
    # (7) coordenação em [0,1] invariante: erosão pela divergência + resposta emergencial
    drho = (1.0 - rho) * (p.r0_rho + p.q_rho * P2) - rho * (p.p_rho * s + p.c_rho)

    return np.array([dF, dL, dM, dD, du, ds, drho])


def _rk4_step(t, x, p, dt, markov_feedback):
    k1 = rhs(t, x, p, markov_feedback)
    k2 = rhs(t + dt / 2, x + dt / 2 * k1, p, markov_feedback)
    k3 = rhs(t + dt / 2, x + dt / 2 * k2, p, markov_feedback)
    k4 = rhs(t + dt, x + dt * k3, p, markov_feedback)
    return project(x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))


# --------------------------------------------------------------------------- #
# Integração (scipy se disponível, senão RK4 de passo fixo em numpy)
# --------------------------------------------------------------------------- #
def integrate(x0, p: Params, t_end=40.0, n_steps=8000, markov_feedback=True):
    """RK4 com projeção no domínio físico (numpy-only, trajetórias limitadas)."""
    x = project(np.asarray(x0, dtype=float))
    h = t_end / (n_steps - 1)
    xs = np.empty((n_steps, x.size))
    xs[0] = x
    for k in range(1, n_steps):
        x = _rk4_step((k - 1) * h, x, p, h, markov_feedback)
        xs[k] = x
    return np.linspace(0.0, t_end, n_steps), xs


# --------------------------------------------------------------------------- #
# Jacobiana, autovalores, pontos fixos e bifurcações
# --------------------------------------------------------------------------- #
def jacobian(x, p: Params, h=1e-6, markov_feedback=True) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.size
    J = np.empty((n, n))
    f0 = rhs(0.0, x, p, markov_feedback)
    for j in range(n):
        dx = np.zeros(n)
        dx[j] = h * max(1.0, abs(x[j]))
        fp = rhs(0.0, x + dx, p, markov_feedback)
        fm = rhs(0.0, x - dx, p, markov_feedback)
        J[:, j] = (fp - fm) / (2 * dx[j])
    return J


def eigen_analysis(x, p: Params, markov_feedback=True):
    J = jacobian(x, p, markov_feedback=markov_feedback)
    vals, vecs = np.linalg.eig(J)
    order = np.argsort(vals.real)[::-1]
    return J, vals[order], vecs[:, order]


def _in_domain(x, tol=1e-6):
    return bool(np.all(x >= _LOWER - tol) and np.all(x <= _UPPER + tol))


def find_fixed_point(p: Params, x0=None, markov_feedback=True):
    """Busca multi-start; retorna o ponto fixo interior de menor resíduo.
    Sem reação fiscal (fiscal_resp=0) a dívida é explosiva para Omega>1 e pode
    não existir ponto fixo interior finito -> o próprio resultado é 'sem equilíbrio'."""
    starts = [x0] if x0 is not None else []
    starts += [
        np.array([0.8, 0.6, 0.5, 1.0, 0.15, 0.10, 0.50]),
        np.array([1.2, 0.3, 0.6, 1.5, 0.30, 0.20, 0.40]),
        np.array([0.5, 1.0, 0.4, 0.8, 0.10, 0.05, 0.60]),
        np.array([1.5, 0.2, 0.7, 2.0, 0.45, 0.30, 0.30]),
    ]
    best, best_res = None, np.inf
    for s0 in starts:
        x = project(np.asarray(s0, dtype=float))
        # Newton amortecido COM projeção (respeita o domínio físico; evita overflow)
        for _ in range(400):
            f = rhs(0.0, x, p, markov_feedback)
            nf = np.linalg.norm(f)
            if nf < 1e-10:
                break
            J = jacobian(x, p, markov_feedback=markov_feedback)
            try:
                step = np.linalg.solve(J, -f)
            except np.linalg.LinAlgError:
                break
            # backtracking para garantir decréscimo do resíduo
            lam = 1.0
            for _ in range(20):
                xn = project(x + lam * step)
                if np.linalg.norm(rhs(0.0, xn, p, markov_feedback)) < nf:
                    break
                lam *= 0.5
            x = xn
        xf = x
        res = float(np.linalg.norm(rhs(0.0, xf, p, markov_feedback)))
        # polimento opcional com scipy a partir de um ponto já no domínio
        if _HAVE_SCIPY and res < 1e-3:
            sol = root(lambda z: rhs(0.0, z, p, markov_feedback), xf, method="hybr",
                       options={"xtol": 1e-12})
            if _in_domain(sol.x) and np.linalg.norm(sol.fun) < res:
                xf, res = sol.x, float(np.linalg.norm(sol.fun))
        if _in_domain(xf) and np.isfinite(res) and res < best_res:
            best, best_res = xf, res
    if best is None:
        return project(starts[0]), False, float("inf")
    return best, best_res < 1e-6, best_res


def bifurcation_margins(x, p: Params) -> dict:
    """Distâncias assinadas aos limiares; >0 significa regime de crise ativado."""
    idx = indices(x, p)
    return dict(
        dPhi=idx["Phi"] - p.Phi_star,
        dOmega=idx["Omega"] - p.Omega_star,
        dSigma=idx["Sigma"] - p.Sigma_star,
        debt_sustainability=idx["Omega"] - 1.0,  # Omega=1: bifurcação transcrítica da dívida
    )


def bifurcation_scan(p: Params, param: str, values):
    """Recomputa o ponto fixo para cada valor do parâmetro e rastreia autovalores.
    Retorna dict com parte real líder, parte real do par complexo (Hopf quando
    cruza 0), nº de direções instáveis e índices no ponto fixo."""
    rows = []
    for v in values:
        pv = Params(**{**asdict(p), param: float(v)})
        xf, ok, res = find_fixed_point(pv)
        if not ok:
            rows.append(dict(val=v, ok=False))
            continue
        _, vals, _ = eigen_analysis(xf, pv)
        cplx = vals[np.abs(vals.imag) > 1e-6]
        cplx_real = float(cplx.real.max()) if cplx.size else float("nan")
        idx = indices(xf, pv)
        rows.append(dict(val=v, ok=True, lead=float(vals.real.max()),
                         complex_real=cplx_real,
                         n_unstable=int(np.sum(vals.real > 1e-6)),
                         Phi=idx["Phi"], Omega=idx["Omega"], Sigma=idx["Sigma"]))
    return rows


# --------------------------------------------------------------------------- #
# Termômetro Composto de Transição de Regime
# --------------------------------------------------------------------------- #
def thermometer_from_indices(Phi, Omega, Sigma, dK, p: Params) -> dict:
    """Termômetro a partir de índices explícitos (não recomputa do estado). Permite
    alimentar Phi/Omega/Sigma empíricos REAIS mantendo a mesma lógica de regimes."""
    V = potentials_from_indices(Phi, Omega, Sigma, dK, p) / p.tau_temp
    V = V - V.max()
    e = np.exp(V)
    pi = e / e.sum()
    m = dict(dPhi=Phi - p.Phi_star, dOmega=Omega - p.Omega_star,
             dSigma=Sigma - p.Sigma_star, debt_sustainability=Omega - 1.0)
    zPhi, zOmega, zSigma = max(0.0, m["dPhi"]), max(0.0, m["dOmega"]), max(0.0, m["dSigma"])
    pressure = 1.0 - math.exp(-(zPhi ** 2 + zOmega ** 2 + zSigma ** 2))
    dominant = int(np.argmax(pi)) + 1
    return dict(
        indices=dict(Phi=Phi, Omega=Omega, Sigma=Sigma),
        regime_probs={REGIMES[i + 1]: float(pi[i]) for i in range(4)},
        dominant_regime=dominant,
        dominant_label=REGIMES[dominant],
        transition_pressure=float(pressure),
        margins=m,
        net_worth_growth=float(dK),
    )


def thermometer(x, p: Params) -> dict:
    idx = indices(x, p)
    th = thermometer_from_indices(idx["Phi"], idx["Omega"], idx["Sigma"],
                                  net_worth_growth(x, p), p)
    th["indices"] = idx          # preserva os componentes completos (T, i_sov, ...)
    return th


# --------------------------------------------------------------------------- #
# Simulação acoplada ODE + cadeia de Markov (quebras estruturais endógenas)
# --------------------------------------------------------------------------- #
def simulate_coupled(x0, p: Params, t_end=40.0, dt=0.05, seed=42, markov_feedback=True):
    rng = np.random.default_rng(seed)
    x = project(np.asarray(x0, dtype=float))
    n = int(t_end / dt)
    traj = np.empty((n + 1, x.size))
    regimes = np.empty(n + 1, dtype=int)
    probs = np.empty((n + 1, 4))
    pressure = np.empty(n + 1)
    state = 0  # regime corrente (índice 0..3); começa no atractor dominante
    state = int(np.argmax(stationary_probs(x, p)))
    traj[0], regimes[0] = x, state + 1
    probs[0] = stationary_probs(x, p)
    pressure[0] = thermometer(x, p)["transition_pressure"]
    for k in range(1, n + 1):
        x = _rk4_step((k - 1) * dt, x, p, dt, markov_feedback)
        # salto discreto de regime pela matriz endógena
        P = transition_matrix(x, p)
        state = int(rng.choice(4, p=P[state]))
        traj[k], regimes[k] = x, state + 1
        probs[k] = stationary_probs(x, p)
        pressure[k] = thermometer(x, p)["transition_pressure"]
    return np.linspace(0, t_end, n + 1), traj, regimes, probs, pressure


# --------------------------------------------------------------------------- #
# Dicionário de proxies empíricas (séries reais -> variáveis do modelo)
# --------------------------------------------------------------------------- #
PROXIES = {
    "F": "CapEx acumulado dos hiperescaladores (MSFT/GOOGL/AMZN/META, 10-Q) + "
         "faturamento de equipamentos semicondutores (SEMI Billings) + receita "
         "data-center da NVIDIA; líquido de depreciação. Fonte: SEC EDGAR, SEMI.",
    "L": "Caixa + investimentos de curto prazo + títulos negociáveis das Big Techs "
         "(balanços 10-Q); ativos de money-market funds (Fed Z.1).",
    "M": "Receita de infraestrutura de nuvem (Synergy/Canalys), market share de "
         "cloud, volume de tokens de APIs de IA.",
    "r_L": "FRED DFII10 (TIPS 10A, juro real longo) e DGS10 (Treasury 10A nominal).",
    "pi_NO/nu_info": "Lucros do setor financeiro (BEA NIPA); fatia de volume HFT; "
                     "FRED VIXCLS (VIX); índice MOVE (ICE BofA).",
    "mu": "Margens operacionais por segmento dos hiperescaladores (disclosures 10-Q).",
    "D/i_sov": "FRED GFDEBTN / FYGFDPUN (dívida federal); juros da dívida (BEA); "
               "term premium ACM (Fed NY); DGS10.",
    "u/W_p/W_o": "FRED PRS85006173 (participação salarial nonfarm), UNRATE; "
                 "BLS OES (software devs vs operadores de dados); participação gig.",
    "G_soc": "Gastos de proteção social (BEA); FRED ICSA (pedidos de seguro-desemprego).",
    "s": "ICE DXY; EURUSD/USDCNY; dispersão cross-country de yields 10A; "
         "taxas de câmbio efetivas reais (BIS).",
    "rho": "Swap lines do Fed (H.4.1); coordenação G20/FSB/BIS; sentimento de "
           "comunicados G7 (proxy).",
    "Sigma": "VIX x volatilidade FX x (1 - proxy de coordenação); MOVE; CDS EUA 5A.",
    "Phi": "FCL_IA ~ fluxo de caixa operacional dos hiperescaladores - CapEx; "
           "lambda*L ~ reservas líquidas; numerador r_L*theta_ind*F.",
    "validacao": "FRED BAMLH0A0HYM2 (spread high-yield), SOFR-OIS, CDS soberanos.",
}


def load_fred(series_map: dict, api_key: str | None = None, start="2015-01-01"):
    """Loader opcional de séries do FRED. Requer pandas + fredapi (ou pandas_datareader)
    e uma chave gratuita do FRED. NÃO embute dados 'atuais' fabricados: o usuário
    fornece as observações. series_map ex.: {'r_L':'DFII10','u':'UNRATE',...}."""
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("pandas necessário para load_fred") from e
    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        out = {k: fred.get_series(v, observation_start=start) for k, v in series_map.items()}
        return pd.DataFrame(out)
    except Exception:
        from pandas_datareader import data as pdr
        return pdr.DataReader(list(series_map.values()), "fred", start)


# --------------------------------------------------------------------------- #
# Demonstração / CLI
# --------------------------------------------------------------------------- #
def _demo():
    p = Params.illustrative()
    x0 = np.array([0.60, 0.80, 0.45, 1.10, 0.12, 0.12, 0.55])

    print("=" * 78)
    print("MODELO TAVARES SFC-ODE-MARKOV  |  calibração ILUSTRATIVA (não observada)")
    print("=" * 78)
    print(f"scipy={_HAVE_SCIPY}  matplotlib={_HAVE_MPL}")

    # ponto fixo e estabilidade local
    xf, ok, res = find_fixed_point(p, x0)
    print(f"\nPonto fixo (convergiu={ok}, ||f||={res:.2e}):")
    names = ["F", "L", "M", "D", "u", "s", "rho"]
    for nm, v in zip(names, xf):
        print(f"  {nm:4s} = {v: .4f}")
    J, vals, vecs = eigen_analysis(xf, p)
    print("\nAutovalores da jacobiana no ponto fixo (ordenados por parte real):")
    for v in vals:
        tag = "INSTÁVEL" if v.real > 1e-6 else ("estável" if v.real < -1e-6 else "neutro")
        print(f"  lambda = {v.real: .5f} {v.imag:+.5f}j   [{tag}]")
    n_unstable = int(np.sum(vals.real > 1e-6))
    print(f"  -> {n_unstable} direção(ões) instável(is): ponto fixo tipo sela "
          f"(desequilíbrio estrutural, não equilíbrio geral).")

    # varredura de bifurcação: velocidade algorítmica nu_info
    print("\nVarredura de bifurcação em nu_info (velocidade informacional/algorítmica):")
    print("  nu_info | real líder | par complexo (Hopf se >0) | instáveis | Sigma")
    hopf, fold, prev = None, None, None
    for row in bifurcation_scan(p, "nu_info", np.linspace(0.5, 4.0, 15)):
        if not row["ok"]:
            print(f"  {row['val']:6.2f} | (sem ponto fixo interior)")
            if fold is None:
                fold = row["val"]
            continue
        print(f"  {row['val']:6.2f} | {row['lead']:10.4f} | {row['complex_real']:23.4f} "
              f"| {row['n_unstable']:9d} | {row['Sigma']:.3f}")
        if prev is not None and prev < 0 <= row["complex_real"] and hopf is None:
            hopf = row["val"]
        prev = row["complex_real"]
    if hopf is not None:
        print(f"  -> Hopf do bloco monetário perto de nu_info ≈ {hopf:.2f}")
    else:
        print("  -> par complexo estável na faixa (sem Hopf numérico).")
    if fold is not None:
        print(f"  -> bifurcação SELA-NÓ (fold): ponto fixo interior desaparece em "
              f"nu_info ≈ {fold:.2f} (colapso do equilíbrio de acumulação).")

    # varredura em r_L: taxa longa dirige a ponte dinheiro-máquinas (Phi)
    print("\nVarredura de bifurcação em r_L (juro real longo -> tensão Phi):")
    print("  r_L    | real líder | instáveis | Phi      | Omega    | Sigma")
    for row in bifurcation_scan(p, "r_L", np.linspace(0.005, 0.06, 12)):
        if not row["ok"]:
            print(f"  {row['val']:6.3f} | (sem ponto fixo interior)")
            continue
        print(f"  {row['val']:6.3f} | {row['lead']:10.4f} | {row['n_unstable']:9d} "
              f"| {row['Phi']:8.3f} | {row['Omega']:8.3f} | {row['Sigma']:.3f}")

    # índices e termômetro no estado inicial e no ponto fixo
    for label, xx in [("estado inicial", x0), ("ponto fixo", xf)]:
        th = thermometer(xx, p)
        print(f"\nTermômetro @ {label}:")
        idx = th["indices"]
        print(f"  Phi={idx['Phi']:.3f} (limiar {p.Phi_star})  "
              f"Omega={idx['Omega']:.3f} (limiar {p.Omega_star})  "
              f"Sigma={idx['Sigma']:.3f} (limiar {p.Sigma_star})")
        print(f"  pressão de transição = {th['transition_pressure']:.3f}  "
              f"regime dominante = R{th['dominant_regime']} ({th['dominant_label']})")
        for lbl, pr in th["regime_probs"].items():
            print(f"    P({lbl}) = {pr:.3f}")

    # simulação acoplada
    t, traj, regimes, probs, pressure = simulate_coupled(x0, p, t_end=40.0, dt=0.05)
    print("\nSimulação acoplada ODE+Markov (40 anos, dt=0.05):")
    print(f"  estado final: F={traj[-1,0]:.3f} L={traj[-1,1]:.3f} M={traj[-1,2]:.3f} "
          f"D={traj[-1,3]:.3f} u={traj[-1,4]:.3f} s={traj[-1,5]:.3f} rho={traj[-1,6]:.3f}")
    frac = {REGIMES[i]: float(np.mean(regimes == i)) for i in range(1, 5)}
    print("  fração de tempo em cada atractor:")
    for lbl, f in frac.items():
        print(f"    {lbl}: {f:.1%}")
    print(f"  pressão de transição final = {pressure[-1]:.3f}")

    # retrato de fase opcional
    if _HAVE_MPL:
        fig, ax = plt.subplots(2, 2, figsize=(12, 9))
        ax[0, 0].plot(t, traj[:, 0], label="F (capital ilíquido)")
        ax[0, 0].plot(t, traj[:, 1], label="L (liquidez)")
        ax[0, 0].set_title("Bloco corporativo"); ax[0, 0].legend(); ax[0, 0].set_xlabel("anos")
        ax[0, 1].plot(t, traj[:, 3], label="D (dívida soberana)", color="tab:red")
        ax[0, 1].plot(t, traj[:, 4], label="u (desemprego tecnológico)", color="tab:orange")
        ax[0, 1].set_title("Bloco fiscal-laboral"); ax[0, 1].legend(); ax[0, 1].set_xlabel("anos")
        idx_traj = np.array([indices(x, p)["Phi"] for x in traj])
        om_traj = np.array([indices(x, p)["Omega"] for x in traj])
        sg_traj = np.array([indices(x, p)["Sigma"] for x in traj])
        ax[1, 0].plot(t, idx_traj, label="Phi")
        ax[1, 0].plot(t, om_traj, label="Omega")
        ax[1, 0].plot(t, sg_traj, label="Sigma")
        ax[1, 0].axhline(p.Phi_star, ls="--", c="gray", lw=0.7)
        ax[1, 0].axhline(p.Omega_star, ls="--", c="gray", lw=0.7)
        ax[1, 0].axhline(p.Sigma_star, ls="--", c="gray", lw=0.7)
        ax[1, 0].set_title("Índices de tensão vs limiares"); ax[1, 0].legend(); ax[1, 0].set_xlabel("anos")
        for i in range(1, 5):
            ax[1, 1].plot(t, probs[:, i - 1], label=f"R{i}")
        ax[1, 1].plot(t, pressure, "k--", label="pressão", lw=1)
        ax[1, 1].set_title("Probabilidades de regime (Markov endógeno)")
        ax[1, 1].legend(); ax[1, 1].set_xlabel("anos")
        fig.tight_layout()
        out = "tavares_phase_portrait.png"
        fig.savefig(out, dpi=110)
        print(f"\nRetrato de fase salvo em {out}")

    return p, xf, (t, traj, regimes, probs, pressure)


if __name__ == "__main__":
    _demo()
