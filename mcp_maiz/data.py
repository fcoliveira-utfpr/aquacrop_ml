"""Consultas somente-leitura sobre os CSVs ja gerados pelos notebooks do aquacrop_ml.

Nao recalcula nada — so carrega (com cache em memoria por processo) e filtra os
arquivos que os notebooks 06-10 ja exportam na raiz do repositorio.
"""

import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def _normalizar(txt: str) -> str:
    sem_acento = "".join(
        c for c in unicodedata.normalize("NFKD", str(txt)) if not unicodedata.combining(c)
    )
    return sem_acento.strip().upper()


@lru_cache(maxsize=None)
def _carregar(nome_arquivo: str) -> pd.DataFrame:
    caminho = REPO_ROOT / nome_arquivo
    if not caminho.exists():
        raise FileNotFoundError(
            f"{nome_arquivo} nao encontrado em {REPO_ROOT}. Rode o notebook que o gera antes de consultar esta tool."
        )
    return pd.read_csv(caminho)


def _filtrar_por_coluna(df: pd.DataFrame, coluna: str, valor: str | None) -> pd.DataFrame:
    if valor is None:
        return df
    alvo = _normalizar(valor)
    return df[df[coluna].astype(str).apply(_normalizar) == alvo]


def _to_records(df: pd.DataFrame) -> list[dict]:
    registros = df.to_dict(orient="records")
    for registro in registros:
        for chave, valor in registro.items():
            if isinstance(valor, float) and pd.isna(valor):
                registro[chave] = None
    return registros


# ---------------------------------------------------------------------------
# Semeadura / aptidao (municipios do Oeste do PR, ja simulados)
# ---------------------------------------------------------------------------

def listar_municipios_oeste() -> list[str]:
    df = _carregar("melhor_data_semeadura_por_municipio.csv")
    return sorted(df["municipio"].unique().tolist())


def melhor_data_semeadura(municipio: str, metodo: str = "integrado") -> list[dict]:
    arquivo = {
        "integrado": "melhor_data_semeadura_integrada.csv",
        "topsis": "melhor_data_semeadura_topsis.csv",
    }.get(metodo)
    if arquivo is None:
        raise ValueError("metodo deve ser 'integrado' ou 'topsis'")
    df = _carregar(arquivo)
    resultado = _filtrar_por_coluna(df, "municipio", municipio)
    if resultado.empty:
        raise ValueError(
            f"Municipio '{municipio}' nao encontrado. Use listar_municipios_oeste() para ver as opcoes."
        )
    return _to_records(resultado)


def matriz_risco_municipio(municipio: str, metodo: str = "integrado") -> list[dict]:
    arquivo = {
        "integrado": "matriz_integrada_oeste_pr.csv",
        "topsis": "matriz_topsis_oeste_pr.csv",
    }.get(metodo)
    if arquivo is None:
        raise ValueError("metodo deve ser 'integrado' ou 'topsis'")
    df = _carregar(arquivo)
    resultado = _filtrar_por_coluna(df, "municipio", municipio)
    if resultado.empty:
        raise ValueError(
            f"Municipio '{municipio}' nao encontrado. Use listar_municipios_oeste() para ver as opcoes."
        )
    return _to_records(resultado.sort_values("data_semeadura"))


# ---------------------------------------------------------------------------
# Produtividade prevista (modelo ExtraTrees, ja rodado para os 50 municipios)
# ---------------------------------------------------------------------------

def produtividade_prevista(municipio: str, ano: int | None = None) -> list[dict]:
    df = _carregar("previsoes_oeste_pr.csv")
    resultado = _filtrar_por_coluna(df, "municipio", municipio)
    if resultado.empty:
        raise ValueError(
            f"Municipio '{municipio}' nao encontrado. Use listar_municipios_oeste() para ver as opcoes."
        )
    if ano is not None:
        resultado = resultado[resultado["ano"] == ano]
    return _to_records(resultado.sort_values(["ano", "data_semeadura"]))


def desempenho_modelo_ml(municipio: str | None = None) -> list[dict]:
    df = _carregar("resultados_inferencia.csv")
    resultado = _filtrar_por_coluna(df, "Municipio", municipio) if municipio else df
    if municipio and resultado.empty:
        disponiveis = sorted(df["Municipio"].unique().tolist())
        raise ValueError(f"Municipio '{municipio}' nao esta entre os validados: {disponiveis}")
    return _to_records(resultado)


# ---------------------------------------------------------------------------
# Custos de producao (CONAB, 6 cidades de referencia) e SEAB/DERAL
# ---------------------------------------------------------------------------

def custo_producao(
    cidade: str,
    ano_inicio: int | None = None,
    ano_fim: int | None = None,
    categoria: str | None = None,
    apenas_total: bool = False,
    deflacionado: bool = True,
) -> list[dict]:
    arquivo = "custo_total_pr_deflacionado.csv" if apenas_total else "df_pr_long_deflacionado.csv"
    df = _carregar(arquivo)
    resultado = _filtrar_por_coluna(df, "cidade", cidade)
    if resultado.empty:
        disponiveis = sorted(df["cidade"].unique().tolist())
        raise ValueError(f"Cidade '{cidade}' nao encontrada. Cidades disponiveis: {disponiveis}")
    if ano_inicio is not None:
        resultado = resultado[resultado["ano"] >= ano_inicio]
    if ano_fim is not None:
        resultado = resultado[resultado["ano"] <= ano_fim]
    if categoria is not None:
        alvo = _normalizar(categoria)
        resultado = resultado[
            resultado["categoria_total"].astype(str).apply(_normalizar).eq(alvo)
            | resultado["categoria_item"].astype(str).apply(_normalizar).eq(alvo)
        ]
    colunas = [
        "cidade", "ano", "secao", "item", "categoria_total", "categoria_item",
        "custo_rs_ha" if not deflacionado else "custo_rs_ha_real",
        "custo_rs_60kg" if not deflacionado else "custo_rs_60kg_real",
    ]
    return _to_records(resultado[colunas].sort_values(["ano", "item"]))


def custo_deral_safrinha(safra: str | None = None) -> list[dict]:
    df = _carregar("custo_deral_milho_safrinha.csv")
    if safra is not None:
        alvo = _normalizar(safra)
        df = df[df["safra"].astype(str).apply(_normalizar) == alvo]
        if df.empty:
            raise ValueError(f"Safra '{safra}' nao encontrada em custo_deral_milho_safrinha.csv")
    return _to_records(df.sort_values("ano"))


def listar_categorias_custo() -> dict:
    df = _carregar("df_pr_long_deflacionado.csv")
    return {
        "categorias_total": sorted(df["categoria_total"].dropna().unique().tolist()),
        "categorias_item": sorted(df["categoria_item"].dropna().unique().tolist()),
        "cidades": sorted(df["cidade"].dropna().unique().tolist()),
    }


# ---------------------------------------------------------------------------
# Precos e IPCA
# ---------------------------------------------------------------------------

def preco_milho(ano_inicio: int | None = None, ano_fim: int | None = None, fonte: str = "deral") -> list[dict]:
    if fonte == "deral":
        df = _carregar("precos_milho_pr.csv")
        col_ano = "ano"
    elif fonte == "ipea":
        df = _carregar("ipea_precos_milho_mensal.csv")
        col_ano = "ano"
    else:
        raise ValueError("fonte deve ser 'deral' ou 'ipea'")
    if ano_inicio is not None:
        df = df[df[col_ano] >= ano_inicio]
    if ano_fim is not None:
        df = df[df[col_ano] <= ano_fim]
    return _to_records(df.sort_values(col_ano))


def fator_deflator_ipca(ano: int, ano_base: int = 2025) -> dict:
    df = _carregar("ipca_anual.csv").set_index("ano")["indice_medio"]
    if ano not in df.index:
        raise ValueError(f"Ano {ano} nao disponivel em ipca_anual.csv (intervalo {df.index.min()}-{df.index.max()})")
    if ano_base not in df.index:
        raise ValueError(f"Ano-base {ano_base} nao disponivel em ipca_anual.csv")
    fator = df[ano_base] / df[ano]
    return {"ano": ano, "ano_base": ano_base, "fator_deflator": fator}
