# Grundrisse — Tavares SFC

Modelo quantitativo da economia política de **Maria da Conceição Tavares** (1992):
financeirização, concorrência schumpeteriana, descarte do trabalho vivo e crise do Estado.
Sistema dinâmico **não-linear** com consistência estoque-fluxo (**SFC**) acoplado a uma
matriz de transição de **Markov endógena**, medindo para qual de quatro atratores de crise
a economia global tende dado o grau de alavancagem financeira e o ciclo de investimento em IA.

> Ferramenta de **cenário / alerta precoce**, não um preditor validado em tempo real.
> A calibração do modelo é **ilustrativa**; os índices empíricos (Φ, Ω, Σ) usam dados reais
> com procedência explícita. Nada é fabricado: série ausente é omitida e reportada.

## Os quatro atratores
- **R1** — Centralização oligopolista por queima de capital
- **R2** — Crise sistêmica monetária e soberana
- **R3** — Barbárie social e paralisia de governabilidade
- **R4** — Reconfiguração estatal top-down

## Simulador no ar (GitHub Pages)
**https://euvinicios.github.io/Grundrisse/**

Abra o link: a página raiz redireciona para o dashboard interativo (`tavares_dashboard.html`).
Funciona offline — sem CDN, sem servidor, gráficos em SVG puro.

### Como usar o dashboard
Duas abas:

1. **Dados reais** — séries Φ/Ω/Σ (2011–2024) com linhas de limiar e badges de procedência.
   - Seletor de **ano** → termômetro empírico (probabilidade de cada regime + gauge de pressão + regime dominante).
   - Card do **snapshot corrente (2024)** e tabela das leituras parciais recentes do boom de Φ (SEC).
2. **Simulação do modelo** — rode o sistema ODE+Markov **ao vivo no navegador**:
   - Ajuste os **sliders** (r_L, ν_info, θ_ind, λ, δ, g1, fiscal_resp, ε, a_u, limiares ilustrativos).
   - Edite o **estado inicial** (F, L, M, D, u, s, ρ), **seed**, `t_end`, `dt`.
   - Clique **Rodar** → gráficos F/L, D/u, índices vs limiares, probabilidades de regime no tempo,
     fração de tempo por atractor. **Reset** restaura os defaults.
   - Painel de **ponto fixo / autovalores / bifurcações** (baseline pré-computado em Python, rotulado como tal).

**Duas leituras separadas, sem misturar escalas:** *Dados reais* usa limiares data-driven reais;
*Simulação do modelo* usa limiares ilustrativos. O mesmo ano de 2024 dá R3 sob limiares ilustrativos
e R1 sob limiares reais — a separação está explícita no rodapé.

## Rodar localmente
- **Duplo-clique** em `tavares_dashboard.html` (funciona via `file://`), ou
- `python -m http.server` e abra `http://localhost:8000/tavares_dashboard.html`.

## CLI (Python)
Use o Python do venv (`.venv/bin/python`) ou o do sistema com numpy/scipy/matplotlib/pandas.

```bash
python tavares_sfc.py            # núcleo: ponto fixo, autovalores, bifurcações, simulação + PNG
python tavares_data.py           # demo empírico (Φ SEC, Ω BM, Σ VIX, limiares, termômetro)
python tavares_data.py --export  # regenera dashboard_data.js a partir das fontes reais
```

## Fidelidade do port JS
O núcleo Python (`tavares_sfc.py`) é a fonte de verdade; `tavares_model.js` é o port para o navegador.
O portão de fidelidade compara os dois:

```bash
node test_model.js   # 10/10 PASS — diffs <= 4.4e-16; trajetória idêntica nos 801 pontos
```

## Fontes de dados (sem chave)
| Índice | Fonte | Cobertura | Notas |
|---|---|---|---|
| **Φ** (ponte dinheiro-máquinas) | SEC EDGAR XBRL (CapEx/treasury hyperscalers) | 2011–2025 | observado |
| **Ω** (tolerância fiscal) | Banco Mundial + Tesouro EUA (10A) − inflação BM | 2000–2024 | 2022+ proxy de juro real documentado por ano |
| **Σ** (risco sistêmico) | CBOE VIX diário → (VIX/100)² | 1990–2026 | ν_info, ρ ilustrativos; **componente vol-FX ausente** |

Bloqueados neste ambiente: FRED, DBnomics; BIS devolve 406. Limiares Φ\*/Ω\*/Σ\* são **escolhas de
quantil (90%)**, não constantes naturais.

## Arquivos
- `tavares_sfc.py` — núcleo: estado 7D, SFC, ODE+Markov, Jacobiano/autovalores, bifurcações, termômetro.
- `tavares_data.py` — camada empírica: loaders WB/SEC/VIX/Tesouro, índices reais, limiares, exportador.
- `tavares_model.js` — port fiel do núcleo para o navegador (sem dependências).
- `test_model.js` — portão de fidelidade JS↔Python (node).
- `tavares_dashboard.html` — dashboard interativo autocontido (SVG vanilla, PT-BR).
- `dashboard_data.js` — dados gerados pelo exportador (`var DATA = {...}`).
- `index.html` — redireciona a raiz do Pages para o dashboard.
- `tavares_phase_portrait.png` — retrato de fase estático gerado pelo núcleo.
