"""Servidor MCP para o aquacrop_ml.

Fase 1: consultas sobre os CSVs ja gerados pelos notebooks (instantaneo, sem rede).
Fase 2: predicao de produtividade "ao vivo" para qualquer municipio do Brasil
(mais lenta — chama NASA POWER e roda o modelo ExtraTrees sob demanda).
"""

from mcp.server.fastmcp import FastMCP

import data
import pipeline

mcp = FastMCP("aquacrop_ml")


# --- Fase 1: semeadura / aptidao ------------------------------------------

@mcp.tool()
def listar_municipios_oeste() -> list[str]:
    """Lista os 50 municipios da Mesorregiao Oeste do Parana ja cobertos pela simulacao (matriz de risco/produtividade/economia pre-calculada)."""
    return data.listar_municipios_oeste()


@mcp.tool()
def melhor_data_semeadura(municipio: str, metodo: str = "integrado") -> list[dict]:
    """Melhor data de semeadura do milho 2a safra recomendada para um municipio do Oeste do PR.

    metodo="integrado": soma ordinal de tercis (clima + produtividade + economia).
    metodo="topsis": PCA (clima+produtividade) + TOPSIS com margem economica — resolve a dupla
    contagem do componente hidrico, recomendado como resposta mais robusta.
    """
    return data.melhor_data_semeadura(municipio, metodo)


@mcp.tool()
def matriz_risco_municipio(municipio: str, metodo: str = "integrado") -> list[dict]:
    """Compara as 6 datas de semeadura possiveis para um municipio (risco climatico, produtividade prevista e margem economica de cada uma)."""
    return data.matriz_risco_municipio(municipio, metodo)


# --- Fase 1: produtividade prevista ----------------------------------------

@mcp.tool()
def produtividade_prevista(municipio: str, ano: int | None = None) -> list[dict]:
    """Produtividade de milho prevista pelo modelo ExtraTrees (kg/ha) para um municipio do Oeste do PR, ja calculada para 1999-2025 x 6 datas de semeadura."""
    return data.produtividade_prevista(municipio, ano)


@mcp.tool()
def desempenho_modelo_ml(municipio: str | None = None) -> list[dict]:
    """Metricas de erro (MAE/MSE/RMSE/R2/MAPE) do modelo ExtraTrees nos municipios usados para validacao (Ceu Azul, Medianeira, Mercedes, Palotina)."""
    return data.desempenho_modelo_ml(municipio)


# --- Fase 1: custos e precos -------------------------------------------------

@mcp.tool()
def custo_producao(
    cidade: str,
    ano_inicio: int | None = None,
    ano_fim: int | None = None,
    categoria: str | None = None,
    apenas_total: bool = False,
    deflacionado: bool = True,
) -> list[dict]:
    """Custo de producao do milho 2a safra (fonte CONAB) para uma das 6 cidades de referencia do PR: Londrina, Campo Mourao, Ubirata, Assis Chateaubriand, Francisco Beltrao, M. Candido Rondon.

    `categoria` filtra por categoria de total (ex. "Custo Total (J)") ou de item (ex. "Fertilizantes") —
    use listar_categorias_custo() para ver as opcoes. `apenas_total=True` retorna so o Custo Total (J) por ano.
    `deflacionado=True` retorna valores em R$ de 2025; False retorna valores nominais da epoca.
    """
    return data.custo_producao(cidade, ano_inicio, ano_fim, categoria, apenas_total, deflacionado)


@mcp.tool()
def custo_deral_safrinha(safra: str | None = None) -> list[dict]:
    """Custo de referencia SEAB/DERAL para milho safrinha (1999/00-2024/25) — a serie usada pelos notebooks 09/10 para a viabilidade economica (substituiu a CONAB por incompatibilidade de escala de produtividade)."""
    return data.custo_deral_safrinha(safra)


@mcp.tool()
def listar_categorias_custo() -> dict:
    """Lista as categorias de custo (total e item) e as cidades disponiveis para filtrar em custo_producao()."""
    return data.listar_categorias_custo()


@mcp.tool()
def preco_milho(ano_inicio: int | None = None, ano_fim: int | None = None, fonte: str = "deral") -> list[dict]:
    """Serie de preco do milho recebido pelo agricultor no Parana. fonte="deral" (anual, nominal+real) ou "ipea" (mensal, 3 series: recebido/atacado PR/atacado SP)."""
    return data.preco_milho(ano_inicio, ano_fim, fonte)


@mcp.tool()
def fator_deflator_ipca(ano: int, ano_base: int = 2025) -> dict:
    """Fator de deflacao IPCA entre `ano` e `ano_base` (multiplique um valor nominal de `ano` por esse fator para obter em R$ de `ano_base`)."""
    return data.fator_deflator_ipca(ano, ano_base)


# --- Fase 2: predicao ao vivo -----------------------------------------------

@mcp.tool()
def prever_produtividade_customizada(municipio: str, ano: int, data_semeadura: str = "15/03") -> dict:
    """Roda a simulacao completa (clima NASA POWER + balanco hidrico FAO + modelo ExtraTrees) para QUALQUER municipio do Brasil, nao so os 50 do Oeste do PR ja pre-calculados.

    Mais lenta que as outras tools (chama uma API externa e pode levar dezenas de segundos) e
    baixa o modelo (~55MB) na primeira chamada. `data_semeadura` precisa ser uma das 6 datas em
    que o modelo foi treinado: 05/02, 15/02, 25/02, 05/03, 15/03, 25/03. Prefira produtividade_prevista()
    quando o municipio ja estiver entre os 50 do Oeste do PR — e instantanea.
    """
    return pipeline.prever_produtividade(municipio, ano, data_semeadura)


if __name__ == "__main__":
    mcp.run()
