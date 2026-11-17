"""
Camada empírica do modelo Tavares (tavares_sfc): dados reais -> índices -> limiares.

Dois loaders, ambos SEM chave de API:
  * fetch_fred()      : fredgraph.csv do FRED (alta frequência: VIX, TIPS, spreads).
                        Fonte primária do dicionário de proxies. Pode estar
                        bloqueada em alguns ambientes; o parser é testado à parte.
  * fetch_worldbank() : API v2 do Banco Mundial (anual, cross-country). Alcançável
                        e usada aqui para demonstrar o pipeline com dados REAIS.

HONESTIDADE METODOLÓGICA (preservada de tavares_sfc):
  - Nada é fabricado. Se uma série não é baixada, ela é marcamente AUSENTE, não
    preenchida com um número "atual" inventado.
  - Cada índice computado carrega um campo de PROCEDÊNCIA dizendo quais insumos
    são observados e quais são parâmetros ilustrativos do modelo.
  - Os limiares Phi*, Omega*, Sigma* NÃO são constantes naturais: são estimados
    por quantis históricos dos próprios índices (data-driven) e sua escolha é
    reportada. A análise de sensibilidade mostra o quanto o diagnóstico depende
    dessa escolha.

Normalização real -> modelo (bloco fiscal-laboral, o que domina a crise R3):
    u     = SL.UEM.TOTL.ZS / 100      desemprego total (proxy de descarte do trabalho)
    D     = GC.DOD.TOTL.GD.ZS / 100   dívida do governo central / PIB
    T     = GC.TAX.TOTL.GD.ZS / 100   arrecadação tributária / PIB (base real)
    i_sov = max(0, FR.INR.RINR)/100 + spread ilustrativo   juro real longo
    G_soc = g0 + g1*u                 reação fiscal ilustrativa ao u REAL
    Omega = (G_soc + G_infra + i_sov*D) / T
    s     = variância cross-country de FR.INR.RINR (divergência monetária real)
    Sigma = nu_info * s * (1 - rho)   nu_info, rho ilustrativos (VIX/swap-lines = FRED)
    Phi   = do estado do modelo (requer CapEx/tesouraria 10-Q + FRED; não há no BM)
"""

from __future__ import annotations

import csv
import io
import json
import urllib.request
import urllib.error
from dataclasses import asdict

import numpy as np

import tavares_sfc as ts

try:
    import pandas as pd
    _HAVE_PD = True
except Exception:  # pragma: no cover
    _HAVE_PD = False

_UA = {"User-Agent": "Mozilla/5.0 (tavares-sfc empirical layer)"}


# --------------------------------------------------------------------------- #
# Loaders (sem chave)
# --------------------------------------------------------------------------- #
def _get(url: str, timeout: float = 30.0, attempts: int = 3) -> bytes:
    """GET com retry/backoff. A API do Banco Mundial é lenta e oscila; um timeout
    isolado não deve zerar o painel. Esgotadas as tentativas, propaga o erro."""
    last = None
    for k in range(attempts):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001 - reporta o último erro real
            last = e
    raise last


def fetch_fred(series_id: str, start="2000-01-01", end=None, timeout=20.0):
    """Baixa uma série do FRED via fredgraph.csv (sem chave). Devolve pd.Series
    indexada por data (ou lista de (data, valor) se pandas ausente). '.' = missing.
    Levanta RuntimeError se a fonte estiver inalcançável (não fabrica dados)."""
    url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
           f"&cosd={start}")
    if end:
        url += f"&coed={end}"
    raw = _get(url, timeout, attempts=1).decode("utf-8", "replace")
    rows = list(csv.reader(io.StringIO(raw)))
    if not rows or rows[0][0].upper() not in ("DATE", "OBSERVATION_DATE"):
        raise RuntimeError(f"FRED: resposta inesperada para {series_id}")
    obs = [(r[0], float(r[1])) for r in rows[1:]
           if len(r) >= 2 and r[1] not in (".", "", "NA")]
    if not obs:
        raise RuntimeError(f"FRED: série {series_id} vazia")
    if _HAVE_PD:
        idx = pd.to_datetime([d for d, _ in obs])
        return pd.Series([v for _, v in obs], index=idx, name=series_id)
    return obs


def fetch_worldbank(country: str, indicator: str, start=2000, end=2024, timeout=20.0):
    """API v2 do Banco Mundial. Devolve dict {ano:int -> valor:float}, sem None."""
    url = (f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
           f"?format=json&date={start}:{end}&per_page=1000")
    data = json.loads(_get(url, timeout).decode("utf-8", "replace"))
    if not isinstance(data, list) or len(data) < 2 or not data[1]:
        raise RuntimeError(f"Banco Mundial: sem dados {country}/{indicator}")
    out = {}
    for rec in data[1]:
        if rec.get("value") is not None:
            out[int(rec["date"])] = float(rec["value"])
    return out


# Indicadores do Banco Mundial usados no bloco fiscal-laboral / divergência.
WB_INDICATORS = {
    "u":     "SL.UEM.TOTL.ZS",     # desemprego (% força de trabalho)
    "D":     "GC.DOD.TOTL.GD.ZS",  # dívida governo central (% PIB)
    "T":     "GC.TAX.TOTL.GD.ZS",  # arrecadação tributária (% PIB)
    "r":     "FR.INR.RINR",        # juro real (%)
    "gdp":   "NY.GDP.MKTP.CD",     # PIB nominal (US$) — contexto
}
WB_COUNTRIES = ["USA", "CHN", "EMU"]  # polo emissor, rival tech, bloco europeu


def wb_panel(countries=None, indicators=None, start=2000, end=2024, timeout=20.0):
    """Painel {pais: {chave: {ano: valor}}}. Tolerante: séries/países que falham
    são omitidos e reportados, jamais preenchidos com valores inventados."""
    countries = countries or WB_COUNTRIES
    indicators = indicators or WB_INDICATORS
    panel, missing = {}, []
    for c in countries:
        panel[c] = {}
        for key, ind in indicators.items():
            try:
                panel[c][key] = fetch_worldbank(c, ind, start, end, timeout)
            except Exception as e:
                missing.append(f"{c}/{key} ({type(e).__name__})")
    return panel, missing


# --------------------------------------------------------------------------- #
# CBOE VIX (Sigma real) e Tesouro EUA (extensão honesta do juro real -> Omega)
# --------------------------------------------------------------------------- #
# O proxy antigo de Sigma (variância anual do juro real entre 2 polos) é esparso e
# de escala ~1e-4, degenerado para fixar limiar. O VIX diário da CBOE (sem chave)
# dá uma variância implícita de equity OBSERVADA e não degenerada. O juro real do
# BM (FR.INR.RINR) para em 2021; o Tesouro EUA (10A nominal, sem chave) menos a
# inflação (BM CPI) estende Omega a 2024 com um proxy ex-post documentado por ano.
VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"


def fetch_vix(timeout=25.0):
    """VIX diário da CBOE (sem chave). Devolve lista [(ano_civil, close)]."""
    raw = _get(VIX_URL, timeout).decode("utf-8", "replace")
    rows = list(csv.reader(io.StringIO(raw)))
    if not rows or rows[0][0].strip().upper() != "DATE":
        raise RuntimeError("CBOE VIX: resposta inesperada")
    hdr = [h.strip().upper() for h in rows[0]]
    if "CLOSE" not in hdr:
        raise RuntimeError("CBOE VIX: coluna CLOSE ausente")
    j = hdr.index("CLOSE")
    out = []
    for r in rows[1:]:
        if len(r) <= j:
            continue
        d = r[0].strip()
        try:
            parts = d.split("/")
            yr = int(parts[2]) if len(parts) == 3 else int(d[:4])
            v = float(r[j])
        except (ValueError, IndexError):
            continue
        out.append((yr, v))
    if not out:
        raise RuntimeError("CBOE VIX: sem observações")
    return out


def vix_annual(daily):
    """{ano: {mean, max, n}} a partir do VIX diário."""
    by = {}
    for yr, v in daily:
        by.setdefault(yr, []).append(v)
    return {y: dict(mean=float(np.mean(a)), max=float(np.max(a)), n=len(a))
            for y, a in sorted(by.items())}


def real_sigma_history(vix_ann, p: ts.Params, rho=0.60):
    """Sigma REAL = nu_info*(VIX_medio_anual/100)^2*(1-rho).

    (VIX/100)^2 é a variância implícita anualizada de equity (VIX é vol em % a.a.).
    nu_info e rho são ILUSTRATIVOS; o componente de volatilidade FX está AUSENTE
    (não há vol-FX observada aqui). Procedência reportada à parte."""
    years = sorted(vix_ann)
    sig = [p.nu_info * (vix_ann[y]["mean"] / 100.0) ** 2 * (1.0 - rho) for y in years]
    return np.array(years), np.array(sig)


def fetch_treasury_10y(year: int, timeout=25.0):
    """Média anual do yield nominal 10A (Tesouro EUA, CSV diário, sem chave), em %."""
    url = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
           f"daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
           f"&field_tdr_date_value={year}&page&_format=csv")
    raw = _get(url, timeout).decode("utf-8", "replace")
    rows = list(csv.reader(io.StringIO(raw)))
    if not rows:
        raise RuntimeError("Tesouro: CSV vazio")
    hdr = [h.strip().strip('"') for h in rows[0]]
    if "10 Yr" not in hdr:
        raise RuntimeError("Tesouro: coluna '10 Yr' ausente")
    j = hdr.index("10 Yr")
    vals = []
    for r in rows[1:]:
        if len(r) > j and r[j].strip() not in ("", "N/A", "NA"):
            try:
                vals.append(float(r[j]))
            except ValueError:
                pass
    if not vals:
        raise RuntimeError(f"Tesouro: sem valores 10Y em {year}")
    return float(np.mean(vals))


def extend_usa_real_rate(panel, start=2000, end=2024, timeout=25.0):
    """Estende o juro real USA além de 2021 (onde FR.INR.RINR para) com um proxy
    ex-post: max(0, 10A nominal Tesouro - inflação BM CPI), em %. Injeta no painel
    e devolve (r_estendido, src_por_ano, treasury_ok). Tolerante: ano sem Tesouro ou
    sem inflação fica AUSENTE (não inventado). A procedência por ano é explícita."""
    usa = panel.setdefault("USA", {})
    r = dict(usa.get("r", {}))
    src = {y: "juro real OBSERVADO (BM FR.INR.RINR)" for y in r}
    try:
        infl = fetch_worldbank("USA", "FP.CPI.TOTL.ZG", start, end, timeout)
    except Exception:  # noqa: BLE001
        infl = {}
    treasury_ok = True
    for y in range(start, end + 1):
        if y in r or y not in infl:
            continue
        try:
            nom = fetch_treasury_10y(y, timeout)
        except Exception:  # noqa: BLE001
            treasury_ok = False
            continue
        r[y] = max(0.0, nom - infl[y])
        src[y] = f"proxy ex-post: 10A Tesouro {nom:.2f}% - inflação BM {infl[y]:.2f}%"
    usa["r"] = r
    return r, src, treasury_ok


# --------------------------------------------------------------------------- #
# SEC EDGAR: capital fixo / tesouraria / fluxo de caixa livre dos hiperescaladores
# --------------------------------------------------------------------------- #
# Fecha o índice Phi com dados corporativos REAIS (XBRL, sem chave). A SEC exige
# User-Agent descritivo com contato (fair access). Estoques (instant) vs fluxos
# (duration ~1 ano) são extraídos por data-fim do período e alinhados por ano civil.
SEC_UA = {"User-Agent": "Grundrisse research tavares-sfc contact research@example.org"}
SEC_COMPANIES = {"MSFT": "0000789019", "GOOGL": "0001652044",
                 "AMZN": "0001018724", "META": "0001326801"}
# tags us-gaap com fallbacks. Empresas TROCAM de tag ao longo do tempo (ex.: AMZN/META/
# GOOGL migraram PP&E para o tag com finance leases ~2018-19; AMZN CapEx migrou para
# ProductiveAssets ~2016). Por isso fundimos a união por ano entre candidatos, em vez de
# escolher um único tag — senão perdemos justamente os anos recentes (boom de CapEx IA).
SEC_TAGS = {
    "F":   (["PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
             "PropertyPlantAndEquipmentNet"], "instant"),                 # capital fixo líquido (incl. finance leases)
    "cash": (["CashAndCashEquivalentsAtCarryingValue",
              "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"], "instant"),
    "sti": (["ShortTermInvestments", "MarketableSecuritiesCurrent",
             "AvailableForSaleSecuritiesDebtSecuritiesCurrent"], "instant"),
    "ocf": (["NetCashProvidedByUsedInOperatingActivities",
             "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"], "duration"),
    "capex": (["PaymentsToAcquirePropertyPlantAndEquipment",
               "PaymentsToAcquireProductiveAssets"], "duration"),
}


def _get_sec(url: str, timeout: float = 25.0, attempts: int = 3) -> bytes:
    last = None
    for _ in range(attempts):
        try:
            req = urllib.request.Request(url, headers=SEC_UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
    raise last


def fetch_sec_annual(cik: str, tag: str, kind: str, timeout=25.0):
    """Série anual {ano_civil: valor_USD} de um tag XBRL. Deduplica por data-fim
    (10-K traz entradas repetidas por ano fiscal do filing). Fluxos: duração ~1 ano."""
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
    d = json.loads(_get_sec(url, timeout).decode("utf-8", "replace"))
    out = {}
    for e in d.get("units", {}).get("USD", []):
        if e.get("form") != "10-K" or e.get("fp") != "FY":
            continue
        end = e.get("end")
        if not end:
            continue
        if kind == "instant":
            if e.get("start") is not None:
                continue
        else:  # duration ~1 ano
            st = e.get("start")
            if not st:
                continue
            from datetime import date
            days = (date.fromisoformat(end) - date.fromisoformat(st)).days
            if not (300 <= days <= 400):
                continue
        out.setdefault(int(end[:4]), float(e["val"]))   # 1º valor por data-fim
    return out


def _merge_tags(cik, candidates, kind, timeout):
    """Une as séries anuais de TODOS os tags candidatos (união por ano; o primeiro
    candidato que tem valor para um ano vence). Cobre trocas de tag ao longo do tempo."""
    merged, used = {}, []
    for tag in candidates:
        try:
            s = fetch_sec_annual(cik, tag, kind, timeout)
        except Exception:  # noqa: BLE001 - tenta o próximo tag
            continue
        if not s:
            continue
        used.append(tag)
        for y, v in s.items():
            merged.setdefault(y, v)
    return merged, used


def sec_panel(companies=None, timeout=25.0):
    """Painel {empresa: {F,cash,sti,ocf,capex: {ano: valor}}} + tags usados + ausentes."""
    companies = companies or SEC_COMPANIES
    panel, used, missing = {}, {}, []
    for name, cik in companies.items():
        panel[name], used[name] = {}, {}
        for key, (cands, kind) in SEC_TAGS.items():
            s, tags = _merge_tags(cik, cands, kind, timeout)
            panel[name][key] = s
            used[name][key] = tags
            if not s:
                missing.append(f"{name}/{key}")
    return panel, used, missing


def sec_aggregate(panel):
    """Agrega F, L(=cash+sti), FCL(=ocf-capex) por ano civil entre empresas.
    Ano válido se >=3 empresas têm F, L e FCL. Devolve {ano: {F,L,FCL,n}}."""
    names = list(panel)
    years = set()
    for nm in names:
        years |= set(panel[nm].get("F", {}))
    agg = {}
    for y in sorted(years):
        F = L = FCL = 0.0
        n = 0
        for nm in names:
            d = panel[nm]
            f = d.get("F", {}).get(y)
            cash = d.get("cash", {}).get(y)
            sti = d.get("sti", {}).get(y, 0.0)
            ocf = d.get("ocf", {}).get(y)
            capex = d.get("capex", {}).get(y)
            if f is None or cash is None or ocf is None or capex is None:
                continue
            F += f
            L += cash + (sti or 0.0)
            FCL += ocf - capex
            n += 1
        if n >= 3:
            agg[y] = dict(F=F, L=L, FCL=FCL, n=n)
    return agg


def real_phi_history(panel_sec, p: ts.Params):
    """Phi REAL por ano: r_L*theta_ind*F / (FCL_IA + lambda*L), com F,L,FCL em US$
    reais (SEC). r_L, theta_ind, lambda são parâmetros do modelo (ilustrativos);
    o conteúdo corporativo (estoques/fluxos) é observado. Unidades cancelam no rácio."""
    agg = sec_aggregate(panel_sec)
    years, phi, cov = [], [], []
    for y, d in agg.items():
        denom = d["FCL"] + p.lam * d["L"]
        if denom <= 0:
            continue
        ph = p.r_L * p.theta_ind * d["F"] / denom
        years.append(y); phi.append(ph); cov.append(d["n"])
    return np.array(years), np.array(phi), cov, agg


# --------------------------------------------------------------------------- #
# Normalização transparente: dados reais -> índices do modelo
# --------------------------------------------------------------------------- #
def _variance(vals):
    a = np.array([v for v in vals if v is not None], dtype=float)
    return float(a.var()) if a.size >= 2 else None


def _omega_inputs(panel, p: ts.Params, y: int, rho: float):
    """Insumos do bloco fiscal Omega para o ano y, a partir só dos EUA (u,D,T,r).
    DESACOPLADO de Sigma: Omega não exige variância cross-country. O juro real r
    pode ser observado (BM, até 2021) ou proxy ex-post (Tesouro-inflação, 2022+),
    já injetado no painel por extend_usa_real_rate. None se faltar u,D,T,r."""
    usa = panel.get("USA", {})
    if not all(y in usa.get(k, {}) for k in ("u", "D", "T", "r")):
        return None
    u = usa["u"][y] / 100.0
    D = usa["D"][y] / 100.0
    T = usa["T"][y] / 100.0
    if T <= 1e-6:
        return None
    r_real = max(0.0, usa["r"][y]) / 100.0
    i_sov = r_real + p.sp0 + p.sp_rho * (1.0 - rho)   # spread ilustrativo
    G_soc = p.g0 + p.g1 * u                            # reação ilustrativa ao u real
    Om = (G_soc + p.G_infra + i_sov * D) / T
    return dict(Omega=Om, u=u, D=D, T=T, i_sov=i_sov, r_real=r_real, G_soc=G_soc)


def _cross_country_s(panel, y):
    """Variância cross-country do juro real (diagnóstico secundário de divergência).
    None se <2 polos com r no ano. NÃO é mais usado para fixar Sigma (ver VIX)."""
    rates = [panel.get(c, {}).get("r", {}).get(y) for c in panel]
    return _variance([None if v is None else v / 100.0 for v in rates])


def real_index_history(panel, p: ts.Params, rho=0.60, sigma_by_year=None):
    """Séries históricas de Omega e Sigma a partir de insumos REAIS.

    Omega usa u, D, T, juro real (EUA; BM até 2021, proxy Tesouro-inflação 2022+);
    G_soc/G_infra/spread são reação fiscal ilustrativa. Sigma vem do VIX (CBOE) via
    sigma_by_year quando fornecido; senão cai no proxy antigo (variância cross-country
    do juro real) apenas para não quebrar chamadas legadas. Devolve (anos, omega,
    sigma, proveniência). Omega e Sigma são alinhados aos anos de Omega."""
    usa = panel.get("USA", {})
    years = sorted(set(usa.get("u", {})) & set(usa.get("D", {}))
                   & set(usa.get("T", {})) & set(usa.get("r", {})))
    omega, sigma, used = [], [], []
    for y in years:
        oi = _omega_inputs(panel, p, y, rho)
        if oi is None:
            continue
        omega.append(oi["Omega"]); used.append(y)
        if sigma_by_year is not None:
            sigma.append(float(sigma_by_year.get(y, float("nan"))))
        else:
            s = _cross_country_s(panel, y)
            sigma.append(p.nu_info * s * (1.0 - rho) if s is not None else float("nan"))
    prov = {
        "Omega": "u,D,T OBSERVADOS (BM/USA); juro real BM até 2021, proxy Tesouro-inflação 2022+; "
                 "G_soc,G_infra,spread ILUSTRATIVOS",
        "Sigma": "VIX médio anual OBSERVADO (CBOE); (VIX/100)^2=variância implícita; "
                 "nu_info,rho ILUSTRATIVOS; vol-FX AUSENTE" if sigma_by_year is not None
                 else "s=variância cross-country do juro real (BM); nu_info,rho ILUSTRATIVOS",
        "Phi": "capital fixo/tesouraria/FCL OBSERVADOS (SEC EDGAR); r_L,theta,lambda ILUSTRATIVOS",
        "rho_coord": rho,
    }
    return np.array(used), np.array(omega), np.array(sigma), prov


def estimate_thresholds(p: ts.Params, omega_hist, sigma_hist=None, quantile=0.90,
                        phi_hist_real=None, sigma_hist_real=None):
    """Estima limiares por quantil, com procedência explícita:

      Omega* : quantil do histórico REAL (BM, juro real estendido a 2024).
      Phi*   : quantil do histórico REAL (SEC EDGAR) se phi_hist_real fornecido;
               senão quantil de trajetória simulada (fallback).
      Sigma* : quantil do histórico REAL de Sigma (VIX/CBOE) se sigma_hist_real
               fornecido e NÃO degenerado (>=10 obs e max>1e-3); senão quantil de
               trajetória SIMULADA. O proxy antigo (variância do juro real entre 2
               polos, ~1e-4) era degenerado; o VIX dá variância implícita observada.

    Devolve Params + dicionário de procedência. A escolha do quantil é reportada."""
    q = float(quantile)
    Om_star = float(np.quantile(omega_hist, q)) if len(omega_hist) else p.Omega_star
    x0 = np.array([0.60, 0.80, 0.45, 1.10, 0.12, 0.12, 0.55])
    _, traj, _, _, _ = ts.simulate_coupled(x0, p, t_end=40.0, dt=0.1)
    sig_hist = np.array([ts.indices(x, p)["Sigma"] for x in traj])
    if (sigma_hist_real is not None and len(sigma_hist_real) >= 10
            and float(np.nanmax(sigma_hist_real)) > 1e-3):
        Sig_star = float(np.quantile(sigma_hist_real, q))
        sig_src = f"quantil {q:.0%} de Sigma REAL (VIX/CBOE)"
    else:
        Sig_star = float(np.quantile(sig_hist, q))
        sig_src = "quantil de trajetória SIMULADA (proxy real esparso/degenerado)"
    if phi_hist_real is not None and len(phi_hist_real):
        Phi_star = float(np.quantile(phi_hist_real, q))
        phi_src = f"quantil {q:.0%} de Phi REAL (SEC EDGAR)"
    else:
        phi_hist = np.array([ts.indices(x, p)["Phi"] for x in traj])
        Phi_star = float(np.quantile(phi_hist, q))
        phi_src = "quantil de trajetória SIMULADA (modelo; SEC indisponível)"
    new = ts.Params(**{**asdict(p), "Phi_star": Phi_star,
                       "Omega_star": Om_star, "Sigma_star": Sig_star})
    info = {
        "quantile": q,
        "Phi_star": (Phi_star, phi_src),
        "Omega_star": (Om_star, f"quantil {q:.0%} de Omega REAL (BM)"),
        "Sigma_star": (Sig_star, sig_src),
        "n_obs_real": len(omega_hist),
    }
    return new, info


def real_snapshot_indices(panel, p: ts.Params, year=None, rho=0.60,
                          phi_model=None, dK_model=0.0, phi_real=None, sigma_real=None):
    """Índices do snapshot: Omega REAL (BM, juro real estendido) + Sigma REAL (VIX)
    se sigma_real fornecido + Phi REAL (SEC) se phi_real fornecido, senão Phi/dK do
    modelo. Coerente com real_index_history (mesma fórmula), logo o limiar estimado
    se aplica à mesma grandeza. s (variância cross-country) é só diagnóstico."""
    usa = panel.get("USA", {})
    cand = sorted(set(usa.get("u", {})) & set(usa.get("D", {}))
                  & set(usa.get("T", {})) & set(usa.get("r", {})))
    y = year if year is not None else (cand[-1] if cand else None)
    if y is None:
        raise RuntimeError("sem ano com u,D,T,r reais para o snapshot")
    oi = _omega_inputs(panel, p, y, rho)
    if oi is None:
        raise RuntimeError(f"insumos reais insuficientes em {y}")
    Om, u, D = oi["Omega"], oi["u"], oi["D"]
    s = _cross_country_s(panel, y)
    if sigma_real is not None:
        Sig = float(sigma_real); sig_src = "Sigma REAL (VIX/CBOE)"
    elif s is not None:
        Sig = p.nu_info * s * (1.0 - rho); sig_src = "Sigma proxy (var. juro real BM)"
    else:
        Sig = float("nan"); sig_src = "Sigma AUSENTE"
    if phi_real is not None:
        Phi = float(phi_real)
        prov = f"Omega REAL (BM+Tesouro), {sig_src}, Phi REAL (SEC); dK,r_L,theta,lambda do modelo"
    else:
        if phi_model is None:
            xf, _, _ = ts.find_fixed_point(p)
            phi_model = ts.indices(xf, p)["Phi"]
        Phi = float(phi_model)
        prov = f"Omega REAL (BM+Tesouro), {sig_src}; Phi,dK do modelo (ilustrativo)"
    return dict(Phi=Phi, Omega=float(Om), Sigma=float(Sig),
                dK=float(dK_model), year=y, u=u, D=D, s=s, prov=prov, sigma_src=sig_src)


# --------------------------------------------------------------------------- #
# Sensibilidade one-at-a-time
# --------------------------------------------------------------------------- #
def _diag(x, p):
    th = ts.thermometer(x, p)
    return th["transition_pressure"], th["dominant_regime"]


def _diag_idx(idx, p):
    th = ts.thermometer_from_indices(idx["Phi"], idx["Omega"], idx["Sigma"],
                                     idx.get("dK", 0.0), p)
    return th["transition_pressure"], th["dominant_regime"]


def sensitivity_thresholds(idx, p: ts.Params, rel_deltas=(-0.20, -0.10, 0.0, 0.10, 0.20)):
    """Perturba cada limiar (Phi*,Omega*,Sigma*) ±rel sobre ÍNDICES FIXOS (reais) e
    mede pressão/regime. Responde: o diagnóstico depende de onde ponho o limiar
    data-driven? idx = dict(Phi,Omega,Sigma,dK)."""
    base_press, base_reg = _diag_idx(idx, p)
    rows = []
    for name in ("Phi_star", "Omega_star", "Sigma_star"):
        v0 = getattr(p, name)
        for d in rel_deltas:
            pp = ts.Params(**{**asdict(p), name: v0 * (1.0 + d)})
            press, reg = _diag_idx(idx, pp)
            rows.append(dict(param=name, delta=d, value=v0 * (1.0 + d),
                             pressure=press, regime=reg,
                             d_pressure=press - base_press,
                             regime_flip=(reg != base_reg)))
    return dict(base_pressure=base_press, base_regime=base_reg, rows=rows)


def sensitivity_params(x, p: ts.Params,
                       params=("nu_info", "r_L", "g1", "fiscal_resp", "theta_ind"),
                       rel_deltas=(-0.20, -0.10, 0.0, 0.10, 0.20)):
    """Sensibilidade estrutural: perturba parâmetros-chave e mede pressão/regime.
    Complementa bifurcation_scan (que olha autovalores) com o diagnóstico agregado."""
    base_press, base_reg = _diag(x, p)
    rows = []
    for name in params:
        v0 = getattr(p, name)
        for d in rel_deltas:
            pp = ts.Params(**{**asdict(p), name: v0 * (1.0 + d)})
            press, reg = _diag(x, pp)
            rows.append(dict(param=name, delta=d, value=v0 * (1.0 + d),
                             pressure=press, regime=reg,
                             d_pressure=press - base_press,
                             regime_flip=(reg != base_reg)))
    return dict(base_pressure=base_press, base_regime=base_reg, rows=rows)


# --------------------------------------------------------------------------- #
# Demonstração com dados reais
# --------------------------------------------------------------------------- #
def _print_sens(title, res):
    print(f"\n{title}  (base: pressão={res['base_pressure']:.3f}, "
          f"regime R{res['base_regime']})")
    print(f"  {'param':12s} {'delta':>6s} {'valor':>9s} {'pressão':>8s} "
          f"{'Δpress':>8s} {'regime':>7s} {'flip':>5s}")
    for r in res["rows"]:
        print(f"  {r['param']:12s} {r['delta']:+6.0%} {r['value']:9.4f} "
              f"{r['pressure']:8.3f} {r['d_pressure']:+8.3f} "
              f"{'R'+str(r['regime']):>7s} {'sim' if r['regime_flip'] else '—':>5s}")


def demo(start=2000, end=2024, quantile=0.90):
    p = ts.Params.illustrative()
    print("=" * 78)
    print("CAMADA EMPÍRICA TAVARES  |  dados REAIS: Banco Mundial + SEC EDGAR + FRED")
    print("=" * 78)

    # 1) Loader FRED (fonte primária). Reporta alcançabilidade sem fabricar dados.
    print("\n[1] Loader FRED (fredgraph.csv, sem chave):")
    try:
        s = fetch_fred("DFII10", start=f"{start}-01-01", timeout=15.0)
        n = len(s)
        last = s.dropna().iloc[-1] if _HAVE_PD else s[-1][1]
        print(f"    DFII10 (TIPS 10A) OK: {n} obs, último={last:.3f}")
        fred_ok = True
    except Exception as e:
        print(f"    FRED inalcançável neste ambiente ({type(e).__name__}). "
              f"O código roda onde o FRED é acessível.")
        fred_ok = False

    # 2) Painel real do Banco Mundial (alcançável).
    print("\n[2] Painel Banco Mundial (anual, cross-country):")
    panel, missing = wb_panel(start=start, end=end, timeout=25.0)
    for c in panel:
        keys = {k: len(v) for k, v in panel[c].items()}
        print(f"    {c}: {keys}")
    if missing:
        print(f"    ausentes (omitidos, NÃO inventados): {missing}")

    # 2b) Complementos: VIX (CBOE) -> Sigma real; Tesouro-inflação -> Omega até 2024.
    print("\n[2b] Complementos (sem chave): VIX/CBOE + juro real estendido (Tesouro-BM):")
    sig_by_year, vix_ann, sig_years, sig_arr = {}, {}, None, None
    vix_ok = False
    try:
        daily = fetch_vix(timeout=25.0)
        vix_ann = vix_annual(daily)
        sig_years, sig_arr = real_sigma_history(vix_ann, p)
        sig_by_year = {int(y): float(v) for y, v in zip(sig_years, sig_arr)}
        vix_ok = True
        print(f"    VIX OK: {len(daily)} obs diárias, {len(vix_ann)} anos "
              f"({min(vix_ann)}–{max(vix_ann)}). Sigma real: "
              f"min={sig_arr.min():.4f} med={np.median(sig_arr):.4f} max={sig_arr.max():.4f}")
        above = [int(y) for y, v in zip(sig_years, sig_arr)
                 if v > float(np.quantile(sig_arr, quantile))]
        print(f"    anos acima do quantil {quantile:.0%} (crises): {above}")
    except Exception as e:  # noqa: BLE001
        print(f"    VIX inalcançável ({type(e).__name__}); Sigma fica no proxy antigo.")
    r_ext, r_src, treasury_ok = extend_usa_real_rate(panel, start=start, end=end, timeout=25.0)
    ext_years = [y for y in sorted(r_src) if "proxy" in r_src[y]]
    print(f"    juro real USA estendido: anos-proxy {ext_years} "
          f"(Tesouro {'OK' if treasury_ok else 'parcial/indisponível'})")
    for y in ext_years:
        print(f"      {y}: {r_src[y]}")

    # 3) Índices reais históricos + estimação de limiares.
    years, om, sig, prov = real_index_history(panel, p, sigma_by_year=sig_by_year or None)
    print(f"\n[3] Índices reais: {len(years)} anos "
          f"({years.min() if len(years) else '—'}–{years.max() if len(years) else '—'})")
    for k, v in prov.items():
        print(f"    {k}: {v}")
    if len(om):
        print(f"    Omega real: min={om.min():.3f} med={np.median(om):.3f} "
              f"max={om.max():.3f}  último({years[-1]})={om[-1]:.3f}")
        sg = sig[np.isfinite(sig)]
        if len(sg):
            print(f"    Sigma real: min={sg.min():.4f} med={np.median(sg):.4f} "
                  f"max={sg.max():.4f}  último({years[-1]})={sig[-1]:.4f}")

    # 3b) Painel SEC EDGAR -> Phi REAL (capital fixo / tesouraria / FCL).
    print("\n[3b] SEC EDGAR (hiperescaladores, XBRL sem chave):")
    phi_real_hist, phi_by_year, agg_sec = None, {}, {}
    try:
        sp, used, smiss = sec_panel(timeout=25.0)
        yrs_phi, phi_arr, cov, agg = real_phi_history(sp, p)
        agg_sec = agg
        for nm in sp:
            def _rng(key):
                ys = sorted(sp[nm].get(key, {}))
                return f"{ys[0]}–{ys[-1]}({len(ys)})" if ys else "—"
            ntags = sum(len(used[nm].get(k, [])) for k in ("F", "ocf", "capex"))
            print(f"    {nm}: F={_rng('F')} ocf={_rng('ocf')} capex={_rng('capex')} "
                  f"[{ntags} tags fundidos]")
        if smiss:
            print(f"    ausentes (omitidos): {smiss}")
        if len(phi_arr):
            phi_real_hist = phi_arr
            phi_by_year = {int(y): float(v) for y, v in zip(yrs_phi, phi_arr)}
            print(f"    Phi real: {len(phi_arr)} anos ({int(yrs_phi.min())}–{int(yrs_phi.max())}), "
                  f"min={phi_arr.min():.4f} med={np.median(phi_arr):.4f} max={phi_arr.max():.4f}")
            print(f"    cobertura n/ano: {dict(zip(yrs_phi.astype(int), cov))}")
            yl = int(yrs_phi[-1]); d = agg[yl]
            print(f"    {yl}: F=${d['F']/1e9:.0f}B  L=${d['L']/1e9:.0f}B  "
                  f"FCL=${d['FCL']/1e9:.0f}B  (n={d['n']})")
        else:
            print("    sem agregado com >=3 empresas; Phi fica do modelo.")
    except Exception as e:  # noqa: BLE001
        print(f"    SEC inalcançável ({type(e).__name__}); Phi fica do modelo.")

    p_est, info = estimate_thresholds(p, om, sig, quantile=quantile,
                                      phi_hist_real=phi_real_hist, sigma_hist_real=sig_arr)
    print(f"\n[4] Limiares estimados (quantil={info['quantile']:.0%}):")
    for name in ("Phi_star", "Omega_star", "Sigma_star"):
        val, src = info[name]
        print(f"    {name:11s} = {val:.4f}   [{src}]")

    # 4b) Bloco Phi REAL recente (boom de CapEx IA). O SEC vai além do BM anual:
    #     leitura PARCIAL — só Phi é observado; Omega/Sigma reais ainda não cobrem
    #     estes anos, logo NÃO compõem o termômetro completo (sem inventar dados).
    if phi_by_year:
        recent = [y for y in sorted(phi_by_year) if y >= 2019]
        if recent:
            print(f"\n[4b] Bloco Phi REAL recente (boom CapEx IA) — Phi*={p_est.Phi_star:.4f}")
            print("    ano   Phi_real   margem(Phi-Phi*)   F($B)   L($B)  FCL($B)  n   acima?")
            for y in recent:
                ph = phi_by_year[y]; d = agg_sec.get(y, {})
                marg = ph - p_est.Phi_star
                flag = "SIM" if marg > 0 else "—"
                print(f"    {y}  {ph:8.4f}   {marg:+10.4f}      "
                      f"{d.get('F', 0)/1e9:6.0f} {d.get('L', 0)/1e9:6.0f} "
                      f"{d.get('FCL', 0)/1e9:7.0f}  {d.get('n', 0)}   {flag}")
            print("    Nota: Omega real vai a 2024 (juro real estendido Tesouro-BM) e Sigma")
            print("    real (VIX) a 2025; o termômetro composto Φ∩Ω∩Σ alcança 2024. Em 2025")
            print("    só Phi é observado (Ω para em 2024): leitura parcial, sem compor regime.")

    # 4) Snapshot: Omega/Sigma REAIS (BM) + Phi/dK do modelo. Mesma fórmula do
    #    histórico, logo o limiar estimado se aplica à grandeza certa.
    xf, _, _ = ts.find_fixed_point(p)
    phi_model = ts.indices(xf, p)["Phi"]
    dK_model = ts.net_worth_growth(xf, p)
    if len(years):
        # ano comum BM (Omega) + SEC (Phi) para um snapshot multi-fonte coerente
        common = sorted(set(years.astype(int)) & set(phi_by_year))
        y_snap = common[-1] if common else int(years[-1])
        idx = real_snapshot_indices(panel, p, year=y_snap, phi_model=phi_model,
                                    dK_model=dK_model, phi_real=phi_by_year.get(y_snap),
                                    sigma_real=sig_by_year.get(y_snap))
        print(f"\n[5] Snapshot {idx['year']} — {idx['prov']}")
        s_txt = f"{idx['s']:.5f}" if idx['s'] is not None else "—"
        print(f"    u_real={idx['u']:.3f} D_real={idx['D']:.3f} s_cross={s_txt}")
        for label, pp in (("limiares ilustrativos", p), ("limiares estimados", p_est)):
            th = ts.thermometer_from_indices(idx["Phi"], idx["Omega"], idx["Sigma"],
                                             idx["dK"], pp)
            print(f"    [{label}] Phi={idx['Phi']:.4f} Omega={idx['Omega']:.3f} "
                  f"Sigma={idx['Sigma']:.4f}")
            print(f"        pressão={th['transition_pressure']:.3f}  "
                  f"dominante=R{th['dominant_regime']} ({th['dominant_label']})")
        # 5) Sensibilidade dos LIMIARES sobre os índices reais fixos (ano corrente).
        _print_sens("[6] Sensibilidade dos LIMIARES (snapshot corrente)",
                    sensitivity_thresholds(idx, p_est))
        # Ano histórico de maior sobrecarga fiscal REAL -> mostra flips de regime.
        y_stress = int(years[int(np.argmax(om))])
        idx_s = real_snapshot_indices(panel, p, year=y_stress, phi_model=phi_model,
                                      dK_model=dK_model, phi_real=phi_by_year.get(y_stress),
                                      sigma_real=sig_by_year.get(y_stress))
        _print_sens(f"[6b] Sensibilidade dos LIMIARES (estresse histórico {y_stress}, "
                    f"Omega_real={idx_s['Omega']:.3f})",
                    sensitivity_thresholds(idx_s, p_est))
    else:
        idx = None
        print("\n[5] Sem histórico real suficiente para snapshot/sensibilidade de limiares.")

    # 6) Sensibilidade ESTRUTURAL dos parâmetros no estado do modelo (recomputa índices).
    _print_sens("[7] Sensibilidade dos PARÂMETROS (estado do modelo)",
                sensitivity_params(xf, p_est))

    print("\nNota: Omega* (BM+Tesouro), Phi* (SEC EDGAR) e Sigma* (VIX/CBOE) são")
    print("estimados de dados REAIS por quantil. Sigma usa (VIX/100)^2 (variância implícita")
    print("de equity); o componente vol-FX segue AUSENTE. Omega 2022+ usa juro real proxy")
    print("ex-post (10A Tesouro - inflação BM), documentado por ano.")
    print("r_L, theta_ind, lambda, nu_info, rho seguem parâmetros ilustrativos do modelo.")
    print("Diagnóstico é ferramenta de cenário/alerta, não preditor validado em tempo real.")
    return dict(panel=panel, years=years, omega=om, sigma=sig,
                params_est=p_est, thresholds=info, fred_ok=fred_ok,
                phi_real=phi_by_year, vix_ok=vix_ok, treasury_ok=treasury_ok,
                sigma_by_year=sig_by_year, vix_annual=vix_ann, r_src=r_src)


# --------------------------------------------------------------------------- #
# Exportador do dashboard (dados reais + baseline do modelo + refs de fidelidade)
# --------------------------------------------------------------------------- #
def _clean(o):
    """Converte numpy/NaN/inf em tipos JSON nativos (NaN/inf -> null)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o


def _probs_array(th):
    return [th["regime_probs"][ts.REGIMES[i]] for i in range(1, 5)]


def export_dashboard(out="dashboard_data.js", start=2000, end=2024, quantile=0.90):
    """Roda o pipeline empírico completo + baseline do modelo e grava um arquivo JS
    autocontido (`var DATA = {...};`) consumido pelo dashboard offline. Mantém as
    duas leituras separadas: dados reais (limiares data-driven) vs modelo (ilustrativo).
    Nada é fabricado; séries ausentes são omitidas e flagadas em reachability."""
    p = ts.Params.illustrative()
    reach = dict(fred=False, worldbank=False, sec=False, vix=False, treasury=False)
    missing_all = []

    # FRED (reporte de alcançabilidade; não usado nos índices aqui)
    try:
        fetch_fred("DFII10", start=f"{start}-01-01", timeout=15.0)
        reach["fred"] = True
    except Exception:  # noqa: BLE001
        pass

    # Banco Mundial + extensão do juro real
    panel, missing = wb_panel(start=start, end=end, timeout=25.0)
    reach["worldbank"] = bool(panel.get("USA", {}).get("u"))
    missing_all += missing
    r_ext, r_src, treasury_ok = extend_usa_real_rate(panel, start=start, end=end, timeout=25.0)
    reach["treasury"] = treasury_ok

    # VIX -> Sigma real
    sig_by_year, vix_ann, sig_years, sig_arr = {}, {}, [], []
    try:
        daily = fetch_vix(timeout=25.0)
        vix_ann = vix_annual(daily)
        sy, sa = real_sigma_history(vix_ann, p)
        sig_years, sig_arr = sy.tolist(), sa.tolist()
        sig_by_year = {int(y): float(v) for y, v in zip(sy, sa)}
        reach["vix"] = True
    except Exception as e:  # noqa: BLE001
        missing_all.append(f"VIX ({type(e).__name__})")

    # SEC -> Phi real
    phi_by_year, agg_sec, cov_by_year, phi_years, phi_arr = {}, {}, {}, [], []
    try:
        sp, used, smiss = sec_panel(timeout=25.0)
        yp, pa, cov, agg = real_phi_history(sp, p)
        agg_sec = agg
        phi_years, phi_arr = yp.tolist(), pa.tolist()
        phi_by_year = {int(y): float(v) for y, v in zip(yp, pa)}
        cov_by_year = {int(y): int(n) for y, n in zip(yp, cov)}
        reach["sec"] = bool(phi_by_year)
        missing_all += smiss
    except Exception as e:  # noqa: BLE001
        missing_all.append(f"SEC ({type(e).__name__})")

    # Históricos reais + limiares data-driven
    years, om, sig, prov = real_index_history(panel, p, sigma_by_year=sig_by_year or None)
    p_est, info = estimate_thresholds(p, om, sig, quantile=quantile,
                                      phi_hist_real=np.array(phi_arr) if phi_arr else None,
                                      sigma_hist_real=np.array(sig_arr) if sig_arr else None)

    # Baseline do modelo (ilustrativo): ponto fixo, autovalores, bifurcações, simulação
    x0 = np.array([0.60, 0.80, 0.45, 1.10, 0.12, 0.12, 0.55])
    xf, ok, res = ts.find_fixed_point(p, x0)
    _, vals, _ = ts.eigen_analysis(xf, p)
    n_unstable = int(np.sum(vals.real > 1e-6))
    scan_nu = ts.bifurcation_scan(p, "nu_info", np.linspace(0.5, 4.0, 15))
    scan_rl = ts.bifurcation_scan(p, "r_L", np.linspace(0.005, 0.06, 12))
    t_s, traj, regimes, probs, pressure = ts.simulate_coupled(x0, p, t_end=40.0, dt=0.05, seed=42)
    fractions = {ts.REGIMES[i]: float(np.mean(regimes == i)) for i in range(1, 5)}
    t_i, xs_i = ts.integrate(x0, p, t_end=40.0, n_steps=8000)

    # Termômetro empírico por ano comum (Phi SEC ∩ Omega BM ∩ Sigma VIX)
    dK_model = float(ts.net_worth_growth(xf, p))   # ~0 no ponto fixo (ilustrativo)
    common = sorted(set(phi_by_year) & set(years.astype(int)) & set(sig_by_year))
    composite = []
    for y in common:
        oi = _omega_inputs(panel, p, y, 0.60)
        if oi is None:
            continue
        th = ts.thermometer_from_indices(phi_by_year[y], oi["Omega"], sig_by_year[y],
                                         dK_model, p_est)
        composite.append(dict(year=int(y), Phi=phi_by_year[y], Omega=oi["Omega"],
                              Sigma=sig_by_year[y], dK=dK_model, u=oi["u"], D=oi["D"],
                              pressure=th["transition_pressure"], dominant=th["dominant_regime"],
                              probs=_probs_array(th), margins=th["margins"]))

    # Snapshot corrente (último ano comum) + leituras parciais recentes de Phi
    snapshot = None
    if common:
        y_snap = common[-1]
        idx = real_snapshot_indices(panel, p, year=y_snap, phi_model=float(ts.indices(xf, p)["Phi"]),
                                    dK_model=dK_model, phi_real=phi_by_year.get(y_snap),
                                    sigma_real=sig_by_year.get(y_snap))
        th = ts.thermometer_from_indices(idx["Phi"], idx["Omega"], idx["Sigma"], idx["dK"], p_est)
        snapshot = dict(year=int(idx["year"]), Phi=idx["Phi"], Omega=idx["Omega"], Sigma=idx["Sigma"],
                        dK=idx["dK"], u=idx["u"], D=idx["D"], s=idx["s"], prov=idx["prov"],
                        pressure=th["transition_pressure"], dominant=th["dominant_regime"],
                        probs=_probs_array(th), margins=th["margins"])
    recent_phi = []
    for y in sorted(phi_by_year):
        if y < 2019:
            continue
        d = agg_sec.get(y, {})
        recent_phi.append(dict(year=int(y), phi=phi_by_year[y],
                               margin=phi_by_year[y] - p_est.Phi_star,
                               F_B=d.get("F", 0.0) / 1e9, L_B=d.get("L", 0.0) / 1e9,
                               FCL_B=d.get("FCL", 0.0) / 1e9, n=d.get("n", 0),
                               above=bool(phi_by_year[y] > p_est.Phi_star)))

    # Referências de fidelidade para o port JS (parte determinística)
    fp_idx = ts.indices(xf, p)
    fp_th = ts.thermometer(xf, p)
    ref = dict(
        params=asdict(p), x0=x0.tolist(), fp_x=xf.tolist(),
        indices_x0=ts.indices(x0, p), fp_indices=fp_idx,
        fp_pressure=fp_th["transition_pressure"], fp_dominant=fp_th["dominant_regime"],
        fp_probs=_probs_array(fp_th),
        integrate_end=xs_i[-1].tolist(),
        sim_end_x=traj[-1].tolist(), sim_end_probs=probs[-1].tolist(),
        sim_end_pressure=float(pressure[-1]),
    )

    DATA = dict(
        meta=dict(generated_from="tavares_data.export_dashboard", quantile=float(quantile),
                  start=int(start), end=int(end),
                  note="Leituras reais (limiares data-driven) e modelo (ilustrativo) são separadas."),
        real=dict(
            years_omega=years.astype(int).tolist(), omega=om.tolist(),
            years_sigma=sig_years, sigma=sig_arr,
            vix_mean=[vix_ann[y]["mean"] for y in sorted(vix_ann)],
            vix_max=[vix_ann[y]["max"] for y in sorted(vix_ann)],
            vix_years=sorted(vix_ann),
            years_phi=phi_years, phi=phi_arr, cov_n=[cov_by_year.get(y, 0) for y in phi_years],
            agg_sec={str(y): dict(F_B=d["F"] / 1e9, L_B=d["L"] / 1e9, FCL_B=d["FCL"] / 1e9, n=d["n"])
                     for y, d in agg_sec.items()},
            thresholds=dict(Phi_star=p_est.Phi_star, Omega_star=p_est.Omega_star,
                            Sigma_star=p_est.Sigma_star, quantile=float(quantile),
                            provenance={k: info[k][1] for k in ("Phi_star", "Omega_star", "Sigma_star")}),
            composite=composite, snapshot=snapshot, recent_phi=recent_phi,
            r_provenance={str(y): r_src[y] for y in sorted(r_src)},
            index_provenance=prov,
            reachability=reach, missing=missing_all,
        ),
        model=dict(
            params=asdict(p), regimes={str(k): v for k, v in ts.REGIMES.items()},
            proxies=ts.PROXIES,
            thresholds_illustrative=dict(Phi_star=p.Phi_star, Omega_star=p.Omega_star, Sigma_star=p.Sigma_star),
            fixed_point=dict(x=xf.tolist(), ok=bool(ok), res=float(res),
                             names=["F", "L", "M", "D", "u", "s", "rho"]),
            eigenvalues=[dict(real=float(v.real), imag=float(v.imag)) for v in vals],
            n_unstable=n_unstable,
            bifurcation=dict(nu_info=scan_nu, r_L=scan_rl),
            baseline_sim=dict(t=t_s.tolist(), traj=traj.tolist(), regimes=regimes.tolist(),
                              probs=probs.tolist(), pressure=pressure.tolist(),
                              x0=x0.tolist(), t_end=40.0, dt=0.05, seed=42, fractions=fractions),
        ),
        ref=ref,
    )

    payload = "var DATA = " + json.dumps(_clean(DATA), ensure_ascii=False, indent=1) + ";\n"
    with open(out, "w", encoding="utf-8") as f:
        f.write(payload)
    print(f"[export] {out} gravado ({len(payload)} bytes).")
    print(f"  reachability: {reach}")
    th = DATA["real"]["thresholds"]
    print(f"  limiares reais: Phi*={th['Phi_star']:.4f} Omega*={th['Omega_star']:.4f} "
          f"Sigma*={th['Sigma_star']:.4f}")
    if snapshot:
        print(f"  snapshot {snapshot['year']}: Phi={snapshot['Phi']:.4f} "
              f"Omega={snapshot['Omega']:.3f} Sigma={snapshot['Sigma']:.4f} "
              f"-> R{snapshot['dominant']} pressão={snapshot['pressure']:.4f}")
    print(f"  anos compostos (Phi∩Omega∩Sigma): {common}")
    return DATA


if __name__ == "__main__":
    import sys
    if "--export" in sys.argv:
        export_dashboard()
    else:
        demo()
