# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## O que é este repositório

`aquacrop_ml` reúne dois pipelines de dados/ciência de dados independentes, ambos sobre milho no
Paraná, em notebooks Jupyter (`.ipynb`) executados localmente (não é um pacote instalável, não há
build/lint/test — é um repositório de análise/pesquisa):

1. **Predição de produtividade via balanço hídrico + ML** (`01`–`05`, pré-existente, migrado do
   Google Colab).
2. **Análise de custos de produção do milho 2ª safra (CONAB)** (`06` + `07_analise_custos_milho_pr`,
   criado nesta sessão).

Os dois pipelines são independentes — não assuma que um depende do outro.

## Fluxo de trabalho neste repositório (git + Colab em paralelo)

Este repo é editado por **dois caminhos ao mesmo tempo**: localmente aqui (VS Code/Claude Code) e
diretamente pelo usuário no **Google Colab** (os notebooks têm badge "Open in Colab" e o Colab
commita direto no GitHub, ex. commits "Criado usando o Colab"). Antes de commitar/dar push:

- Sempre rode `git fetch` / `git status` primeiro — é comum o `origin/main` ter avançado por commits
  feitos via Colab que não existem no clone local (já aconteceu: um push local foi rejeitado porque
  o Colab tinha adicionado `05_inferencias.ipynb` nesse meio-tempo). Resolver com merge normal
  (`git pull origin main --no-rebase`) costuma bastar, sem conflito, já que os dois lados mexem em
  arquivos diferentes na maior parte das vezes.
- A identidade do git (`user.name`/`user.email`) é configurada **localmente** neste repositório
  (não `--global`) — confirme com o usuário antes de definir/alterar.
- Push por HTTPS pode falhar com `HTTP 408`/timeout ao enviar o `.xls` (~2.5MB) ou notebooks grandes
  com gráficos embutidos; se acontecer, `git config http.postBuffer 524288000` (local, já
  configurado neste repo) resolve — reexecute o `git push` depois.

## Ambiente Python — ATENÇÃO (duas instalações no mesmo PC)

Existem **dois Pythons 3.13 distintos** nesta máquina Windows, e cada notebook pode acabar rodando
em um deles dependendo de como for aberto:

- `C:\Users\fabri\AppData\Local\Programs\Python\Python313\python.exe` — o `python` que o terminal
  (Bash/PowerShell do Claude Code) resolve por padrão.
- `C:\Users\fabri\anaconda3\python.exe` — o interpretador que o **kernel Jupyter do VS Code**
  normalmente seleciona (Anaconda é detectada e preferida pela extensão Jupyter).

Pacotes instalados via `pip install` num terminal **não aparecem automaticamente no outro**
ambiente (já causou `ModuleNotFoundError: xlrd` e depois `ImportError` mesmo após reiniciar o
kernel, porque o restart não troca de interpretador). Ao instalar uma dependência nova:

```bash
pip install <pacote>                                    # ambiente do terminal/Bash
"/c/Users/fabri/anaconda3/python.exe" -m pip install <pacote>   # ambiente do kernel do VS Code
```

Pacotes já confirmados como instalados em **ambos** os ambientes: `pandas`, `numpy`, `matplotlib`,
`scipy`, `xlrd` (necessário para ler `.xls` antigo — `openpyxl` não serve para esse formato),
`ipywidgets`, `nbformat`, `nbclient`, `ipykernel`.

Para rodar um notebook de ponta a ponta sem abrir o VS Code (validação rápida, mesmo ambiente do
terminal):

```bash
python -m nbconvert --to notebook --execute --inplace <notebook>.ipynb
```

## Nota sobre acentuação nos terminais

Ao inspecionar `.ipynb`/`.csv` via Bash neste ambiente, caracteres acentuados (ã, ç, õ...) aparecem
como `�` no terminal. **Isso é só a code page do console** — os arquivos estão em UTF-8 correto (já
verificado byte a byte via `ord()`/`hex()`). Não "conserte" a acentuação nos arquivos achando que
está corrompida; abra em VS Code/Jupyter para ver o texto correto.

---

## Pipeline 1 — Balanço hídrico + Machine Learning (produtividade de milho)

Fluxo sequencial entre os 4 notebooks (cada um consome a saída do anterior):

1. **`01_vies_tecnologico.ipynb`** — remove a tendência de ganho tecnológico da série histórica de
   produtividade (`produtividade_locais.csv`, `produtividade_milho_oeste.csv`, municípios do oeste
   do PR, 2007–2020) via regressão linear, isolando o efeito climático/hídrico da produtividade.
2. **`02_bh_cultura_milho.ipynb`** — calcula o balanço hídrico da cultura (metodologia FAO) por fase
   fenológica, usando dados climáticos (a lista de municípios de referência para extração via
   Google Earth Engine está em `geoinfos.js`; coordenadas/altitude ficam em
   `municipios_inferencia.csv`). Produz features por fase (Tmin/Tmax/Tmed, Chuva, UR, ETc, ETR, ARM,
   DEF, EXC) — ver `df_final.csv`.
3. **`03_feature_engineering.ipynb`** — transforma o formato longo (`df_final.csv`, uma linha por
   fase) em formato largo (`df_wide.csv`, uma linha por ano/safra com colunas sufixadas `_F1`..`_F4`
   por fase), e avalia correlação entre features e `Yield_obs`.
4. **`04_treinamento_ML.ipynb`** — treina e compara RandomForest, XGBoost, LightGBM, ExtraTrees,
   MLP (scikit-learn/xgboost/lightgbm/optuna) para prever produtividade a partir de `df_wide.csv`.
   Exporta os modelos treinados (`RandomForest.joblib`, `XGBoost.joblib`, `LightGBM.joblib`).
5. **`05_inferencias.ipynb`** — carrega os modelos treinados, prepara os dados dos municípios de
   inferência (baixa dados, refaz o feature engineering) e roda a inferência propriamente dita,
   gerando `resultados_inferencia.csv` (MAE/MSE/RMSE/R²/MAPE por município). Chegou ao repo local via
   merge de um commit feito direto no Colab (não foi criado nesta sessão) — ver seção de fluxo de
   trabalho acima.

Os notebooks têm badge "Open in Colab" e trechos com `google.colab`/`!pip install` — foram escritos
originalmente para Colab; ao rodar localmente, células que dependem do Colab (`from google.colab
import files`, uploads interativos) precisam ser adaptadas ou puladas.

---

## Pipeline 2 — Custos de produção do milho 2ª safra no Paraná (CONAB)

Criado nesta sessão a partir de `milho_2a_safra_serie_historica_2005-2025.xls` (série histórica
CONAB de custo de produção, 254 abas — uma por combinação cidade-UF-ano, ex. `Londrina-PR-2020`).

### `06_custos_milho_parana.ipynb` — extração

- Filtra as 85 abas do Paraná (regex `^(.*)-PR-(\d{4})$`), cobrindo 6 cidades: Londrina, Campo
  Mourão, Ubiratã, Assis Chateaubriand, Francisco Beltrão, M. Cândido Rondon.
- **A planilha muda de layout 3 vezes ao longo da série** (4 colunas até ~2018, 13 colunas com
  células mescladas em anos intermediários, 5 colunas limpas em 2025). Ler por índice fixo de coluna
  quebra silenciosamente em parte dos anos. A função `parse_aba_custo` resolve isso de forma
  genérica: localiza a linha/coluna do cabeçalho `DISCRIMINAÇÃO`, e para cada linha de item usa a
  **ordem** dos valores numéricos não nulos à direita do rótulo (1º = custo R$/ha, 2º = custo
  R$/60kg, 3º/4º = participação %) — funciona nos 3 formatos sem tratar cada um manualmente. Ao
  filtrar valores numéricos, é essencial excluir `NaN` explicitamente (`pd.notna`), pois `NaN` é
  `float` e passa num `isinstance(v, (int, float))` ingênuo, embaralhando a ordem posicional.
- Gera `custos_pr[cidade][ano] -> DataFrame` (estrutura pedida pelo usuário) **e** `df_pr_long`
  (formato longo, uma linha por item de custo × cidade × ano — melhor para filtrar por intervalo e
  comparar cidades). Exportado como `df_pr_long.csv`.
- Tem um filtro interativo (`ipywidgets`) para consultar por cidade(s)/intervalo de anos/item.

### `07_analise_custos_milho_pr.ipynb` — análise e correção monetária

Lê `df_pr_long.csv` e faz:

- **Normalização de categorias**: os 158 rótulos de `item` variam de nome entre os 3 layouts.
  `classificar_total` mapeia linhas de total/subtotal para 10 categorias canônicas via
  palavra-chave sem acento (não confiar no código-letra `(A)...(J)` sozinho — há pelo menos um erro
  de digitação na planilha original da CONAB, um ano rotula "Renda de Fatores" como `(F)` em vez de
  `(I)`). `classificar_item` agrupa itens individuais relevantes (fertilizantes, sementes,
  defensivos, mão de obra, arrendamento, transporte, seguro, juros, máquinas/depreciação,
  administração, terra) por palavra-chave, para composição de custos sem mapear as 158 variações.
- **Correção pelo IPCA**: busca ao vivo a série 433 do SGS/Banco Central (variação mensal do IPCA,
  mesma base do IBGE) via `urllib`; se falhar (sem internet), cai para o cache local
  `ipca_mensal.csv`/`ipca_anual.csv` (já commitados no repo). Constrói número-índice (base
  dez/1998=100), calcula índice médio anual, e deflaciona `custo_rs_ha`/`custo_rs_60kg` para **R$ de
  2025** (`fator_deflator = índice_2025 / índice_do_ano`). Limitação assumida: o dataset só tem o
  ano do relatório (não o mês exato tipo "Março/2020"), então usa média anual do IPCA como
  aproximação — se precisar de mais precisão, seria necessário voltar ao `.xls` e capturar o campo
  "Mês/Ano" de cada aba (hoje não está em `df_pr_long.csv`).
  - Ano-base de deflação (2025) está fixado na variável `ANO_BASE` — mude ali se quiser outro ano.
- Estatística descritiva, séries temporais com CAGR real, boxplot entre cidades, composição de
  custos (área empilhada), custo por saca real, correlação custo×produtividade (Pearson/Spearman),
  volatilidade (variação % a.a. e CV).
- Exporta `df_pr_long_deflacionado.csv` e `custo_total_pr_deflacionado.csv`.

### Arquivos de dados deste pipeline

| Arquivo | Conteúdo |
|---|---|
| `milho_2a_safra_serie_historica_2005-2025.xls` | Fonte bruta CONAB (254 abas, todo o Brasil) |
| `df_pr_long.csv` | Custos do PR, formato longo, valores **nominais** |
| `df_pr_long_deflacionado.csv` | Igual ao anterior + colunas `custo_rs_ha_real`/`custo_rs_60kg_real` (R$ de 2025) e categorias normalizadas |
| `custo_total_pr_deflacionado.csv` | Só a linha "Custo Total (J)" por cidade/ano, nominal e real |
| `ipca_mensal.csv` / `ipca_anual.csv` | Cache do IPCA (BCB SGS série 433), usado como fallback offline |

Se o `.xls` fonte for atualizado (nova série histórica da CONAB), reexecute `06_custos_milho_parana.ipynb`
antes de `07_analise_custos_milho_pr.ipynb`, na ordem.

---

## Notebook 10 — Análise integrada (risco climático + produtividade prevista + econômico)

`10_analise_integrada.ipynb` estende a classificação de aptidão do `09_analise_custo_modelos.ipynb`
adicionando a **produtividade prevista pelo ExtraTrees** como um terceiro eixo independente na
recomendação de melhor data de semeadura (antes só clima via ISNA + econômico via DERAL).

- **Não refaz simulação nem inferência** — reaproveita os CSVs já exportados pelo `09`
  (`previsoes_oeste_pr.csv`, `risco_climatico_oeste_pr.csv`, `risco_economico_oeste_pr.csv`). Se o
  `09` for reexecutado com dados novos, reexecute este notebook em seguida.
- **Seções 1–4** — soma ordinal de tercis dos 3 eixos (clima + produtividade + econômico), mesma
  lógica do `09`. Descobriu-se que ISNA e produtividade prevista têm correlação forte (Pearson
  r≈0,85) — o componente hídrico acaba sendo ponderado duas vezes nessa abordagem.
- **Seções 5–6** — resolve a redundância acima com **PCA** (reduz clima+produtividade a um único
  `componente_agroclimatico`, explica ~92,5% da variância) seguido de **TOPSIS** (2 critérios:
  componente agroclimático × margem econômica, pesos 50/50 ajustáveis na variável `pesos`) para
  gerar um `topsis_score` contínuo (0–1) — mais robusto que a soma ordinal por preservar magnitude
  e não contar a água duas vezes. A recomendação de melhor data via TOPSIS muda em relação à do
  `09` (2 eixos) em só 8/50 municípios, contra 35/50 na soma ordinal — o TOPSIS fica bem mais
  próximo do comportamento original por resolver a redundância hídrica.
- **Ambiente local (Mac, diferente do PC Windows descrito acima)**: `.venv/` criado na raiz do repo
  (fora do git, ver `.gitignore`) com `pandas numpy matplotlib scipy joblib scikit-learn requests
  ipywidgets nbformat nbclient ipykernel xlrd`; kernel Jupyter registrado como `aquacrop_ml_venv`
  via `python -m ipykernel install --user --name aquacrop_ml_venv`.

### Arquivos gerados por este notebook

| Arquivo | Conteúdo |
|---|---|
| `matriz_integrada_oeste_pr.csv` | 3 eixos + score ordinal por tercis (município × data de semeadura) |
| `melhor_data_semeadura_integrada.csv` | Melhor data por município via score ordinal |
| `matriz_topsis_oeste_pr.csv` | 3 eixos + `componente_agroclimatico` (PCA) + `topsis_score` |
| `melhor_data_semeadura_topsis.csv` | Melhor data por município via TOPSIS |
