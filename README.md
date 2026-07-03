# aquacrop_ml

Repositório com dois estudos sobre milho no Paraná: um de **predição de produtividade** via balanço
hídrico da cultura + machine learning, e outro de **análise de custos de produção** (2ª safra) a
partir da série histórica da CONAB.

## 1. Predição de produtividade (balanço hídrico + ML)

Pipeline sequencial, migrado do Google Colab, para prever a produtividade do milho no oeste do
Paraná a partir de variáveis climáticas e de balanço hídrico da cultura:

1. **`01_vies_tecnologico.ipynb`** — remove a tendência de ganho tecnológico da série histórica de
   produtividade (regressão linear), isolando o efeito climático.
2. **`02_bh_cultura_milho.ipynb`** — calcula o balanço hídrico da cultura por fase fenológica
   (metodologia FAO), a partir de dados climáticos dos municípios listados em `geoinfos.js`.
3. **`03_feature_engineering.ipynb`** — organiza as features por fase em formato largo (uma linha
   por safra) e avalia a correlação com a produtividade observada.
4. **`04_treinamento_ML.ipynb`** — treina e compara RandomForest, XGBoost, LightGBM, ExtraTrees e
   MLP; exporta os modelos treinados (`.joblib`) e os resultados de inferência por município.

### Dados e modelos

- `produtividade_locais.csv`, `produtividade_milho_oeste.csv` — série histórica de produtividade
  observada (2007–2020) por município.
- `df_final.csv` / `df_wide.csv` — features de balanço hídrico em formato longo e largo.
- `municipios_inferencia.csv`, `df_final_inferencia.csv`, `df_wide_inferencia.csv` — dados dos
  municípios usados na inferência (predição fora da amostra de treino).
- `RandomForest.joblib`, `XGBoost.joblib`, `LightGBM.joblib` — modelos treinados.
- `resultados_inferencia.csv` — métricas de erro (MAE, MSE, RMSE, R², MAPE) por município.

## 2. Análise de custos de produção — milho 2ª safra, Paraná (CONAB)

A partir da série histórica de custos de produção da CONAB (`milho_2a_safra_serie_historica_2005-2025.xls`,
254 abas cobrindo todo o Brasil), este estudo extrai e analisa os custos das cidades do **Paraná**
(Londrina, Campo Mourão, Ubiratã, Assis Chateaubriand, Francisco Beltrão e M. Cândido Rondon).

1. **`06_custos_milho_parana.ipynb`** — lê as 85 abas do Paraná (o layout da planilha muda 3 vezes
   ao longo dos anos; o parser lida com isso automaticamente), monta um dicionário
   `custos_pr[cidade][ano]` e um DataFrame longo `df_pr_long` (uma linha por item de custo × cidade
   × ano), com um filtro interativo por cidade/intervalo de anos. Exporta `df_pr_long.csv`.
2. **`07_analise_custos_milho_pr.ipynb`** — análise completa a partir de `df_pr_long.csv`:
   - Normaliza os rótulos de custo (que variam de nome entre os layouts) em categorias
     comparáveis entre anos.
   - **Corrige os valores pelo IPCA** (série oficial do Banco Central/IBGE, buscada ao vivo com
     fallback para cache local), deflacionando tudo para R$ de 2025 — sem isso, comparar custo
     nominal de 1999 com o de 2025 mistura inflação com custo real.
   - Estatística descritiva, séries temporais com taxa de crescimento real (CAGR), comparação entre
     cidades, composição de custos, custo por saca, correlação custo × produtividade e volatilidade
     ano a ano.
   - Exporta `df_pr_long_deflacionado.csv` e `custo_total_pr_deflacionado.csv`.

### Dados deste estudo

- `ipca_mensal.csv` / `ipca_anual.csv` — cache do índice IPCA (Banco Central, série 433), usado
  quando não há acesso à internet no momento de rodar o notebook.

## Requisitos

Python 3.13 com: `pandas`, `numpy`, `matplotlib`, `scipy`, `xlrd` (leitura do `.xls` antigo),
`ipywidgets`, `nbformat`/`nbclient` (execução de notebooks via linha de comando). Para o pipeline
1: adicionalmente `scikit-learn`, `xgboost`, `lightgbm`, `optuna`, `seaborn`.

## Como rodar um notebook pela linha de comando

```bash
python -m nbconvert --to notebook --execute --inplace <notebook>.ipynb
```
