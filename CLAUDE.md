# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## O que é este repositório

`aquacrop_ml` reúne dois pipelines de dados/ciência de dados independentes, ambos sobre milho no
Paraná, em notebooks Jupyter (`.ipynb`) executados localmente (não é um pacote instalável, não há
build/lint/test — é um repositório de análise/pesquisa):

1. **Predição de produtividade via balanço hídrico + ML** (`01`–`05`, pré-existente, migrado do
   Google Colab).
2. **Análise de custos de produção do milho 2ª safra** (`06`–`10`, incluindo `09b`/`10b`) — a fonte
   de custo evoluiu de CONAB para **DERAL** no meio do trabalho; ver nota de descontinuação na
   seção "Pipeline 2" abaixo antes de mexer em custo.

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

## Pipeline 2 — Custos de produção do milho 2ª safra no Paraná (CONAB) — DESCONTINUADO

> **A fonte de custo da CONAB foi descartada em favor do DERAL** (ver notebook `09`, seção 6): o
> pacote CONAB usado aqui é "Agricultura Empresarial — Alta Tecnologia — OGM", calibrado pra uma
> produtividade (~6.260 kg/ha) muito acima do que a simulação de balanço hídrico atinge, o que
> distorcia a classificação de viabilidade. Os notebooks `06`/`07` e os CSVs abaixo **continuam no
> repo como referência histórica**, mas nenhuma análise nova deve depender deles — use o eixo
> econômico DERAL do notebook `09` (`custo_deral_milho_safrinha.csv`,
> `risco_economico_oeste_pr.csv`).

Criado numa sessão anterior a partir de `milho_2a_safra_serie_historica_2005-2025.xls` (série
histórica CONAB de custo de produção, 254 abas — uma por combinação cidade-UF-ano, ex.
`Londrina-PR-2020`).

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

## Pipeline 2 (atual) — Eixo econômico via DERAL (`08`–`10`)

- **`08_precos.ipynb`** — série de preços do milho (IPEA/SEAB-Deral), usada como receita de
  referência no eixo econômico.
- **`09_analise_custo_modelos.ipynb`, seção 5** — risco climático via **ISNA** (Índice de
  Satisfação da Necessidade de Água, `ETR/ETc`), seguindo o critério oficial de **déficit hídrico**
  do Zoneamento Agrícola de Risco Climático (ZARC), conforme a **Portaria SPA/MAPA nº 329, de
  28/07/2026** (milho 2ª safra, Paraná, safra 2026/2027, item 1.14, critério "a"):
  [PDF oficial](https://www.gov.br/agricultura/pt-br/assuntos/riscos-seguro/programa-nacional-de-zoneamento-agricola-de-risco-climatico/portarias/safra-vigente/parana/PO5FD71.PDF).
  `simular_safra` calcula o ISNA sobre as **fases próprias do ZARC** (Grupo I, ciclo 90-110 dias,
  item 1.12: Fase I = dias 1-20, Fase III = dias 51-80) — diferentes das fases Doorenbos-Kassam
  (F1-F4) usadas pra produtividade atingível. Um ano é hidricamente adverso se ISNA da Fase I ≤ 0,70
  **ou** ISNA da Fase III ≤ 0,55; cada (município, data de semeadura) é classificado pela
  **frequência de anos adversos** nos 27 anos simulados: Risco climático 20%/30%/40%/Inapto (>40%)
  — substituiu a versão anterior (média do ISNA ponderada por `ky`, corte ilustrativo 0,65/0,50).
  - **Critérios (b) frio na floração, (c) geada e (d) excesso hídrico da mesma portaria não entram
    na classificação final** — ficam só como colunas de diagnóstico
    (`freq_frio_floracao`/`freq_geada`/`freq_excesso_faseIV` em `risco_climatico_oeste_pr.csv`). A
    portaria pública não detalha como os 4 critérios se combinam: testamos união (qualquer critério
    dispara → 300/300 "Inapto", com o critério de frio sozinho disparando em 83% dos anos —
    implausível pra uma região onde o milho safrinha é cultivado rotineiramente) e "o critério mais
    restritivo manda" (mais plausível, mas ainda concentrado em "Inapto", 0 combinações no nível
    20%). Sem a definição oficial exata, ficamos só com o critério hídrico (a) pra não arriscar uma
    combinação que pareça mais "oficial" do que realmente é.
  - Resultado com só o critério (a): 181/300 "Inapto", 62 em 40%, 57 em 30%, **0 em 20%** — nenhuma
    combinação atinge o nível de risco mais baixo nesse critério isolado.
  - `df_wide_oeste.csv` precisou ser reconstruído por inteiro (recálculo via NASA POWER API, ~1.350
    combinações município×ano, ~40 min) pra incluir as colunas das fases ZARC — se os limiares/fases
    mudarem de novo, precisa reconstruir de novo.
- **`09_analise_custo_modelos.ipynb`, seção 6** — custo de referência **SEAB/DERAL "Estimativa de
  Custos de Produção — Milho Safrinha"** (`nhistorico_94.xls`, aba `milho_saf`), 26 safras
  (1999/00–2024/25), decomposto em `custo_fixo_real` e `custo_var_kg_real` (deflacionados pelo
  IPCA) → exportado como `custo_deral_milho_safrinha.csv`. A margem por (município, data de
  semeadura, ano) é `receita − (custo_fixo + custo_var_kg × produtividade_prevista)`; o
  `risco_economico` (Menos viável / Viabilidade intermediária / Mais viável) é um **tercil relativo**
  da margem média entre as 300 combinações — não um corte absoluto de viabilidade.
- **Seção 6.1 — sensibilidade ao rateio do custo fixo**: o cenário principal atribui 100% do
  `custo_fixo` (terra, depreciação, administração) ao milho safrinha, como se fosse plantado
  isolado; mas na prática ele **sucede a soja** no mesmo ano-agrícola, na mesma área/máquina, então
  parte desse custo fixo já é amortizada pela cultura principal. A célula 6.1 recalcula tudo com
  `fator_rateio_custo_fixo = 0.5` (`calcular_risco_economico(fator, rotulo)`, reaproveitada pros dois
  cenários) e exporta `risco_economico_oeste_pr_rateio50.csv`, comparando com o cenário de 100%.
  - **Achado**: em termos **absolutos** o efeito é grande — margem média das 300 combinações vai de
    ‑R$898/ha (100% do custo fixo, 0/300 combinações com margem ≥ 0) para +R$3/ha (rateio 50%,
    172/300 com margem ≥ 0), e o % médio de anos que cobrem o custo sobe de 5,8% para 47,7%.
  - Em termos **relativos**, porém, `risco_economico` (tercil) não muda em nenhuma das 300
    combinações — o rateio desconta um valor praticamente constante por ano de todas as
    combinações município×data, então a ordenação relativa entre elas não se altera. Ou seja: o
    rateio muda a leitura de "o milho safrinha se paga sozinho?" (absoluta), mas não muda "qual
    município/data é relativamente melhor que outro?" (o que alimenta a matriz do notebook 10).
  - **Os 3 cenários (100%, 50%, proporcional) entram lado a lado na matriz da seção 7** e em
    `melhor_data_semeadura_por_municipio.csv` (colunas `margem_media_rs_ha_100pct`/`_rateio50`/
    `_rateio_proporcional`) — mas como `risco_economico` (tercil) é idêntico nos 3, o `score_economico`
    e o `score_combinado` não mudam entre cenários; só a margem em R$/ha (contínua) muda. O cenário
    100% continua sendo o único usado pra classificar (`ORDEM_ECONOMICO`), os outros dois ficam como
    referência numérica. Ver seção 6.1 do notebook `10` pra onde essa diferença contínua realmente
    importa (TOPSIS).
- **Seção 6.2 — rateio proporcional aos dias de ocupação**: não há literatura encontrada que
  justifique especificamente 50%; a alternativa mais defensável é ratear o custo fixo proporcional
  ao tempo que cada cultura ocupa a área (é esse tempo de uso que consome depreciação de
  máquina/oportunidade da terra). Com ciclos médios aproximados (soja ~160 dias, milho safrinha
  ~110 dias, ajustáveis em `CICLO_SOJA_DIAS`/`CICLO_MILHO_SAFRINHA_DIAS`), o fator fica em **~41%**
  — exportado em `risco_economico_oeste_pr_rateio_proporcional.csv`. Resultado: margem média das 300
  combinações sobe pra **+R$170/ha**, com **300/300 combinações** cobrindo o custo em média (contra
  0/300 no cenário de 100% e 172/300 no de 50%) — mas, como nos outros cenários, a classificação
  relativa (`risco_economico`, tercil) continua **igual em 100% das combinações**, porque o desconto
  segue sendo aproximadamente uniforme por ano entre município×data.

---

## Notebook 09b — 4º cenário de custo fixo: "Custo Operacional (CONAB)"

`09b_analise_custo_modelos.ipynb` nasceu de uma dúvida sobre a seção 6.1/6.2 do `09`: os cenários
de rateio (50%/proporcional) descontam uma fração **assumida** do `custo_fixo_real` inteiro — mas
esse "custo fixo" (DERAL, `nhistorico_94.xls`) mistura desembolso real (depreciação de máquina, mão
de obra permanente) com custo **imputado** (Renda de Fatores — remuneração de capital próprio e da
terra), sem separar os dois. Investigação em duas partes, ambas reaproveitando dados já no repo:

- **Terra vs. máquina entre culturas** (`custos.xlsx`, aba `so_a_base` — consulta detalhada do site
  SEAB/DERAL, 9 culturas × 5 datas recentes, não usada em nenhum outro notebook antes deste): a
  **Remuneração da terra é idêntica** (R$ 1.491,02/ha) em milho 2ª safra, soja, café, feijão e
  trigo — confirma que o DERAL atribui o custo de oportunidade da terra cheio a cada cultura, sem
  descontar nada pela sucessão soja→milho safrinha na mesma área/ano. Já a depreciação de máquinas
  **varia por cultura** (370 no milho safrinha vs. 410 na soja) — não há duplicação óbvia aí.
- **Proporção Renda de Fatores/Custo Total, ano a ano** (`df_pr_long_deflacionado.csv`, pipeline
  `06`/`07` — a série CONAB descontinuada como referência de custo *absoluto*, mas com a estrutura
  de custo íntegra: tem as categorias `Custo Operacional (H)` e `Total Renda de Fatores (I)` que a
  série DERAL não separa). A proporção agregada por ano parece instável (desvio-padrão 7% entre
  anos) até quebrar por cidade: as 6 cidades do PR sobem/descem **juntas** dentro do mesmo ano
  (desvio entre cidades no mesmo ano, 5,1%, menor que entre anos) — não é ruído de parsing, é
  variação macro real (provavelmente juros — 2021, SELIC baixa da pandemia, teve a proporção mais
  baixa da série, 6,1%; valorização da terra — 2015-2018 e 2023-2025 tiveram as mais altas, até
  36,6%). Validado de duas formas: (1) a coluna `participacao_pct` já calculada na planilha bate com
  o ratio derivado de `custo_rs_ha` até 2010, mas tem um bug de escala a partir de 2011 (fração vs.
  ponto percentual — por isso não é usada); (2) `Custo Operacional (H)` bate com `Custo Total (J) −
  Renda de Fatores (I)` com diferença de R$0,00/ha nos 27 anos.
- **Cenário resultante**: `custo_operacional_real = custo_rs_ha_total_real × (1 − ratio_renda_fatores_conab[ano])`,
  aplicado sobre a série DERAL (a fonte de custo absoluto não muda — só a *proporção* de quanto
  descontar vem do CONAB, por não existir essa quebra na série DERAL). Ao contrário dos cenários
  50%/proporcional (fração fixa em qualquer ano), esse desconto **varia ano a ano** seguindo a
  proporção medida.
- **Resultado**: reexecuta as seções 6-7 do `09` (risco econômico, matriz clima×econômico, melhor
  data, mapa de calor) com esse 4º cenário lado a lado dos outros 3. Mesmo padrão já visto em 09:
  efeito grande em termos absolutos (margem média salta de −R$898/ha no 100% para +R$32/ha no
  Operacional CONAB), mas a classificação relativa (tercil) e a melhor data recomendada por
  município **não mudam em nenhuma combinação** frente ao cenário 100% (0/300 e 0/50).
- **Limitação principal**: a proporção vem do pacote CONAB "Alta Tecnologia" (~104 sc/ha), não do
  próprio pacote DERAL "Milho Safrinha" (~60-80 sc/ha) — um pacote de maior insumo dilui o peso
  relativo da Renda de Fatores no total, então a proporção medida (≈22% em média) pode estar
  levemente subestimada frente ao pacote DERAL real (o snapshot único de `custos.xlsx` sugeriu
  ≈29% pro milho 2ª safra no período mais recente). Não usa o `09` como dependência de execução —
  só reaproveita os CSVs que ele já exporta (`previsoes_oeste_pr.csv`, `risco_climatico_oeste_pr.csv`,
  `risco_economico_oeste_pr*.csv`, `custo_deral_milho_safrinha.csv`, `precos_milho_pr.csv`), mesma
  lógica de reaproveitamento do `10`.

### Arquivos gerados por este notebook

| Arquivo | Conteúdo |
|---|---|
| `risco_economico_oeste_pr_operacional_conab.csv` | Risco econômico (tercil + margem) via cenário Custo Operacional (CONAB) |
| `matriz_risco_climatico_economico_oeste_pr_operacional_conab.csv` | Matriz risco climático × econômico com os 4 cenários de margem lado a lado |
| `melhor_data_semeadura_por_municipio_operacional_conab.csv` | Melhor data por município via esse cenário |
| `custo_deral_operacional_conab.csv` | Série de custo por safra (real/nominal) do cenário Operacional CONAB — mesmo formato de `custo_deral_milho_safrinha.csv`, mas com o fixo já descontado da Renda de Fatores; usada pelo `12` no gráfico de custo/preço |

---

## Notebook 10 — Análise integrada (risco climático + produtividade prevista + econômico)

`10_analise_integrada.ipynb` estende a classificação de aptidão do `09_analise_custo_modelos.ipynb`
adicionando a **produtividade prevista pelo ExtraTrees** como um terceiro eixo independente na
recomendação de melhor data de semeadura (antes só clima via ISNA + econômico via DERAL).

- **Não refaz simulação nem inferência** — reaproveita os CSVs já exportados pelo `09`
  (`previsoes_oeste_pr.csv`, `risco_climatico_oeste_pr.csv`, `risco_economico_oeste_pr.csv`). Se o
  `09` for reexecutado com dados novos, reexecute este notebook em seguida.
- **Seções 1–4** — soma ordinal de tercis dos 3 eixos (clima + produtividade + econômico), mesma
  lógica do `09` (note que o eixo climático agora tem **4 níveis**, não 3 — `ORDEM_CLIMA` mapeia
  Risco climático 20%/30%/40%/Inapto para 0-3, então `score_integrado` vai de 0 a 7, não mais 0 a
  6). Descobriu-se que ISNA e produtividade prevista têm correlação forte (Pearson r≈0,85) — o
  componente hídrico acaba sendo ponderado duas vezes nessa abordagem.
- **Seções 5–6** — resolve a redundância acima com **PCA** (reduz clima+produtividade a um único
  `componente_agroclimatico`, a partir de `isna_medio` — o ISNA médio ponderado por `ky`, não a
  classificação categórica nova — explica ~92,5% da variância) seguido de **TOPSIS** (2 critérios:
  componente agroclimático × margem econômica, pesos 50/50 ajustáveis na variável `pesos`) para
  gerar um `topsis_score` contínuo (0–1) — mais robusto que a soma ordinal por preservar magnitude
  e não contar a água duas vezes.
  - Com a classificação climática oficial do ZARC (seção 5 do `09`, ver acima — bem mais
    concentrada em "Inapto" do que a média ilustrativa antiga), a recomendação de melhor data via
    TOPSIS muda em relação à do `09` (2 eixos) em 42/50 municípios, e a soma ordinal muda em 16/50
    — o próprio TOPSIS não mudou de cálculo (usa `isna_medio` contínuo, não a categoria), mas a
    recomendação do `09` (que usa a categoria) mudou bastante, então a comparação também muda.
- **Seção 6.1 — sensibilidade do TOPSIS ao rateio do custo fixo**: `matriz_integrada` carrega a
  margem dos 3 cenários (`margem_media_rs_ha_100pct`/`_rateio50`/`_rateio_proporcional` — ver 09,
  seções 6.1/6.2), e `calcular_topsis()` é chamada uma vez pra cada, gerando
  `topsis_score`/`topsis_score_rateio50`/`topsis_score_rateio_proporcional`. Diferente do
  `score_economico` ordinal (idêntico nos 3 cenários porque usa só o tercil), o TOPSIS usa a margem
  **contínua** na normalização vetorial e por isso **é sensível ao rateio**: correlação de Spearman
  entre `topsis_score` (100%) e o de 50% é 0,9825 (22/50 municípios com melhor data diferente);
  entre 100% e o proporcional (~41%) é 0,9996 (só 2/50 municípios mudam). Ou seja, o cenário de
  rateio de 50% desloca a recomendação de data bem mais que o proporcional — outro motivo pra
  preferir o proporcional como cenário de referência secundário, já que ele muda menos a
  recomendação em relação ao cenário principal (100%). Exportado em
  `melhor_data_semeadura_topsis_sensibilidade_rateio.csv`. O cenário 100% continua sendo o que
  alimenta o heatmap e a exportação principal (`melhor_data_semeadura_topsis.csv`).
- **Ambiente local (Mac, diferente do PC Windows descrito acima)**: `.venv/` criado na raiz do repo
  (fora do git, ver `.gitignore`) com `pandas numpy matplotlib scipy joblib scikit-learn requests
  ipywidgets nbformat nbclient ipykernel xlrd`; kernel Jupyter registrado como `aquacrop_ml_venv`
  via `python -m ipykernel install --user --name aquacrop_ml_venv`.

### Arquivos gerados por este notebook

| Arquivo | Conteúdo |
|---|---|
| `matriz_integrada_oeste_pr.csv` | 3 eixos + score ordinal por tercis (município × data de semeadura) |
| `melhor_data_semeadura_integrada.csv` | Melhor data por município via score ordinal |
| `matriz_topsis_oeste_pr.csv` | 3 eixos + `componente_agroclimatico` (PCA) + `topsis_score` dos 3 cenários de rateio |
| `melhor_data_semeadura_topsis.csv` | Melhor data por município via TOPSIS (cenário 100% do custo fixo) |
| `melhor_data_semeadura_topsis_sensibilidade_rateio.csv` | Melhor data via TOPSIS nos 3 cenários lado a lado (seção 6.1) |

---

## Notebook 10b — Análise integrada usando só o cenário Custo Operacional (CONAB)

`10b_analise_integrada.ipynb` repete a estrutura do `10` (3 eixos: ISNA + produtividade prevista +
econômico; soma ordinal seções 1-4; PCA+TOPSIS seções 5-6) usando **só** o cenário de custo fixo
"Custo Operacional (CONAB)" do `09b` como eixo econômico — sem a seção 6.1 do `10` (sensibilidade
ao rateio entre 100%/50%/proporcional), já que aqui há só um cenário, nada pra comparar entre
cenários de custo. Resultado igual em estrutura ao `10` (Pearson ISNA×produtividade = 0,849, PCA
explica 92,5% da variância) — só a magnitude da margem econômica (e por consequência o
`topsis_score`) muda, por vir do cenário Operacional (CONAB) em vez do 100%. Depende dos CSVs do
`09` (clima/produtividade) e do `09b` (`risco_economico_oeste_pr_operacional_conab.csv`) — reexecute
se qualquer um dos dois mudar.

### Arquivos gerados por este notebook

| Arquivo | Conteúdo |
|---|---|
| `matriz_integrada_oeste_pr_operacional_conab.csv` | 3 eixos + score ordinal por tercis, cenário Operacional (CONAB) |
| `melhor_data_semeadura_integrada_operacional_conab.csv` | Melhor data por município via score ordinal |
| `matriz_topsis_oeste_pr_operacional_conab.csv` | 3 eixos + `componente_agroclimatico` (PCA) + `topsis_score`, cenário único |
| `melhor_data_semeadura_topsis_operacional_conab.csv` | Melhor data por município via TOPSIS |

---

## Servidor MCP (`mcp_maiz/`) e consulta sem código (`11_assistente_perguntas.ipynb`)

Chegaram ao repo via commit direto no GitHub (não criados nesta sessão) — expõem os resultados dos
notebooks 06-10 como *tools* consultáveis, sem precisar abrir nenhum notebook.

- **`mcp_maiz/data.py`** — camada **somente-leitura**: `_carregar()` só faz `pd.read_csv` num dos
  CSVs que os notebooks 06-10 já exportam na raiz do repo (`melhor_data_semeadura_*.csv`,
  `matriz_*_oeste_pr.csv`, `custo_deral_milho_safrinha.csv`, `precos_milho_pr.csv`, etc.), com
  `@lru_cache(maxsize=None)` por nome de arquivo. Não recalcula nada e não tem nenhuma lógica
  amarrada aos rótulos/categorias específicos das colunas (ex. os nomes das classes de
  `risco_climatico`) — só filtra e devolve registros. **Por isso mudanças de metodologia nos
  notebooks (ex. a reclassificação do risco climático via ZARC, seção 5 do `09` acima) não exigem
  nenhuma alteração de código aqui**, só reexecutar os notebooks pra regerar os CSVs.
  - **Ressalva de cache**: se o servidor MCP estiver rodando como processo persistente (ex. via
    Claude Desktop/IDE), o `lru_cache` mantém os DataFrames antigos em memória mesmo depois dos
    CSVs serem regenerados — **precisa reiniciar o processo** do servidor pra ele enxergar dados
    novos. Não é um problema imediato aqui no Mac: o `.mcp.json` aponta pro Python do Windows
    (`C:\Users\fabri\...`), então o servidor não roda nesta máquina.
- **`mcp_maiz/pipeline.py`** — predição de produtividade "ao vivo" (`prever_produtividade_customizada`,
  fase 2) pra **qualquer** município do Brasil, não só os 50 do Oeste do PR pré-calculados. É uma
  **cópia adaptada e independente** da simulação de balanço hídrico do `09` (células 1, 3, 5, 7, 8,
  10, 12 — ver docstring do arquivo), reescrita como funções puras (sem loop de 50 municípios × 27
  anos, roda 1 combinação por vez). **Não inclui as fases/critérios do ZARC** adicionados nesta
  sessão — só o balanço hídrico Doorenbos-Kassam (F1-F4) e a produtividade via ExtraTrees, porque
  seu único propósito é prever produtividade, nunca fez classificação de risco climático. Continua
  correto sem alteração, mas é uma **duplicação de código que pode dessincronizar** se
  `simular_safra` do `09` mudar de novo em algo que afete a produtividade (ex. `DURACAO_CICLO`, Kc,
  ky, definição das fases Doorenbos-Kassam) — nesse caso, `pipeline.py` precisaria ser atualizado
  manualmente também.
- **`11_assistente_perguntas.ipynb`** — formulário Colab (`ipywidgets`) que importa direto
  `mcp_maiz/data.py`/`mcp_maiz/pipeline.py` como funções Python (não via protocolo MCP) pra dar uma
  interface de menu, sem texto livre nem chave de API, pra pesquisadores consultarem os mesmos
  dados. Roda como processo novo a cada execução no Colab, então não sofre do problema de cache do
  item acima.
