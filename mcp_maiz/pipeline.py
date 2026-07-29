"""Predicao de produtividade "ao vivo" para um municipio qualquer do Brasil.

Extraido e adaptado de 09_analise_custo_modelos.ipynb (celulas 1, 3, 5, 7, 8, 10, 12) —
mesma fisica (balanco hidrico FAO por fase fenologica) e o mesmo modelo (ExtraTrees)
usados la, mas reescrito como funcoes puras reutilizaveis (sem estado de notebook, sem
loop de 50 municipios/27 anos: roda sob demanda para 1 municipio/1 ano/1 data por vez).

Diferenca deliberada em relacao ao notebook: `buscar_clima_diario` recebe `altitude`
como parametro (no 09 ela vinha de uma variavel global `alt_atual` setada no loop
externo, o que so funciona dentro daquele loop).
"""

import unicodedata
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.exceptions import InconsistentVersionWarning

# Filtro global, definido uma unica vez na importacao (thread-safe: warnings.filters e
# process-wide e catch_warnings() nao e thread-safe para mutacao concorrente — o SDK do
# MCP abre um catch_warnings(record=True) ao redor de cada requisicao, entao qualquer
# warning emitido durante uma tool call (ex. mismatch de versao do sklearn ao carregar o
# .joblib) precisa estar filtrado de antemao, nunca via catch_warnings aninhado em runtime).
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

_REDE_EXECUTOR = ThreadPoolExecutor(max_workers=4)


def _com_timeout(func, *args, timeout, **kwargs):
    """Roda func numa thread separada com um teto de tempo real.

    `requests.get(timeout=...)` nao cobre resolucao de DNS de forma confiavel em todo
    ambiente (ja observado neste projeto: um hang de dezenas de minutos sem timeout do
    requests disparar, sem nenhuma conexao TCP sequer chegar a se abrir). Isso garante
    que a tool sempre retorna, mesmo que a chamada de rede em si fique presa para sempre.
    """
    future = _REDE_EXECUTOR.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError as exc:
        raise TimeoutError(
            f"{func.__name__} nao respondeu em {timeout}s — verifique a conexao de rede."
        ) from exc

URL_DADOS_CULTURAS = "https://raw.githubusercontent.com/fcoliveira-utfpr/agrometeorologia/refs/heads/main/dados_culturas.csv"
URL_MODELO_EXTRATREES = "https://github.com/fcoliveira-utfpr/aquacrop_ml/releases/download/6/ExtraTrees.joblib"

SEM = ["05/02/", "15/02/", "25/02/", "05/03/", "15/03/", "25/03/"]
DATAS_SEMEADURA_SUPORTADAS = [d.rstrip("/") for d in SEM]

DURACAO_CICLO = 100
CIAFMAX = 0.5
CC = (0.35 + 0.45) / 2
U = (10 + 13) / 2
Z_INICIAL = 0.05

FEATURES = [
    "UR_F1", "T_DEF_F1", "DEF_F1", "Tmax_F1", "Tmed_F2", "Tmed_F1",
    "Tmax_F2", "UR_F2", "ISNA_F1", "Tmin_F2", "Tmin_F3", "T_DEF_F2",
    "ETc_F1", "DEF_F2", "Altitude_F1", "Amp_F4", "Tmin_F1", "Pefetiva_F1",
    "PA_F2", "Tmed_F3", "Chuva_F1", "Tmin_F4", "ISNA_F2", "PA_F3", "EXC_F1",
    "Longitude_F1", "PA_F4", "Amp_F3", "Chuva_F2", "ARM_F1", "ETR_F4",
    "EXC_F2", "ETR_F2", "Pefetiva_F2", "ARM_F3", "Tmed_F4", "ARM_F2",
    "Tmax_F3", "ETc_F2", "ETR_F1", "Amp_F1", "Latitude_F1", "ETR_F3",
    "UR_F4", "T_DEF_F4", "DEF_F4", "ISNA_F3", "Pefetiva_F4", "ISNA_F4",
    "UR_F3", "Tmax_F4", "lat_lon_F1", "EXC_F3", "ETc_F3", "ETc_F4", "PA_F1",
    "Chuva_F3", "ARM_F4", "DEF_F3", "Chuva_F4", "Amp_F2", "EXC_F4",
    "Pefetiva_F3", "T_DEF_F3", "DOY_semeadura", "DOY_maturacao",
]

VARS_FASE = [
    "Tmin", "Tmax", "Tmed", "Chuva", "UR", "ETc", "ETR", "ARM", "DEF", "EXC",
    "ISNA", "PA", "Amp", "Pefetiva", "Latitude", "Longitude", "Altitude", "T_DEF", "lat_lon",
]


def _sem_acento(txt: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(txt)) if not unicodedata.combining(c)).upper()


# ---------------------------------------------------------------------------
# Parametros da cultura (milho) e localizacao do municipio
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _parametros_milho() -> dict:
    cache = CACHE_DIR / "dados_culturas.csv"
    try:
        dados = _com_timeout(pd.read_csv, URL_DADOS_CULTURAS, timeout=30)
        dados.to_csv(cache, index=False)
    except Exception:
        if not cache.exists():
            raise RuntimeError(
                "Sem internet e sem cache local de dados_culturas.csv — nao e possivel montar os parametros da cultura."
            )
        dados = pd.read_csv(cache)

    for c in dados.columns:
        if c != "Cultura":
            dados[c] = dados[c].astype(str).str.replace(",", ".", regex=False)
            dados[c] = pd.to_numeric(dados[c], errors="coerce")

    milho = dados[dados["Cultura"] == "Milho"].iloc[0]

    fase1, fase2, fase3, fase4 = milho["F1 %"], milho["F2 %"], milho["F3 %"], milho["F4 %"]
    fase_total = fase1 + fase2 + fase3 + fase4
    fase1d = (DURACAO_CICLO * fase1) / fase_total
    fase2d = (DURACAO_CICLO * fase2) / fase_total
    fase3d = (DURACAO_CICLO * fase3) / fase_total
    fase4d = (DURACAO_CICLO * fase4) / fase_total

    kc1, kc2, kc3 = milho["Kc ini"], milho["Kc méd"], milho["Kc fin"]
    z_final = milho["Z efetivo (m)"]

    return {
        "KY1": milho["ky1"], "KY2": milho["ky2"], "KY3": milho["ky3"], "KY4": milho["ky4"],
        "KC1": kc1, "KC2": kc2, "KC3": kc3,
        "FASE1D": fase1d, "FASE2D": fase2d, "FASE3D": fase3d, "FASE4D": fase4d,
        "ACRESCIMO_KC1": (kc2 - kc1) / fase2d,
        "ACRESCIMO_KC2": (kc3 - kc2) / fase4d,
        "Z_FINAL": z_final,
        "ACRESCIMO_Z": (z_final - Z_INICIAL) / (fase1d + fase2d),
    }


def buscar_municipio(nome: str) -> dict:
    """Localiza lat/lon/altitude/DTA (capacidade de agua do solo) em clima_solo_local.csv (5563 municipios do Brasil)."""
    solo = pd.read_csv(REPO_ROOT / "clima_solo_local.csv")
    solo["_norm"] = solo["Município"].apply(_sem_acento)
    alvo = _sem_acento(nome)
    achado = solo[solo["_norm"] == alvo]
    if achado.empty:
        parecidos = solo[solo["_norm"].str.contains(alvo, na=False)]["Município"].head(5).tolist()
        dica = f" Voce quis dizer: {parecidos}?" if parecidos else ""
        raise ValueError(f"Municipio '{nome}' nao encontrado em clima_solo_local.csv.{dica}")
    row = achado.iloc[0]
    dta = float(str(row["DTA (mm/m)"]).replace(",", "."))
    return {
        "municipio": row["Município"],
        "estado": row["Estado"],
        "altitude": float(row["Altitude"]),
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "dta": dta,
    }


# ---------------------------------------------------------------------------
# Balanco hidrico FAO (identico a 09_analise_custo_modelos.ipynb, celulas 7-8)
# ---------------------------------------------------------------------------

def _dec_sol(nda):
    return 23.45 * np.sin(np.deg2rad(360 / 365 * (nda - 80)))


def _hora_nascer_sol(d, lat):
    return np.rad2deg(np.arccos(-(np.tan(np.deg2rad(lat)) * np.tan(np.deg2rad(d)))))


def _e_saturacao(temp):
    return 0.6108 * 10 ** ((7.5 * temp) / (237.3 + temp))


def _bol(tmax, tmin, ea, qg, qgcs):
    a = 4.903e-9 * (((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2)
    b = 0.34 - 0.14 * np.sqrt(ea)
    c = 1.35 * (qg / qgcs) - 0.35
    return -(a * b * c)


def _eto_penman(s, rn, gama, u2, es, ea, tmed):
    eto1 = 0.408 * s * rn
    eto2 = (gama * 900 * u2 * (es - ea)) / (tmed + 273)
    eto3 = s + gama * (1 + 0.34 * u2)
    return (eto1 + eto2) / eto3


def buscar_clima_diario(latitude: float, longitude: float, altitude: float, ano_inicial: int) -> pd.DataFrame:
    """Baixa clima diario (NASA POWER) e calcula ETo (Penman-Monteith) para 2 anos-calendario a partir de ano_inicial."""
    ano_final = ano_inicial + 1
    ini = int(datetime(ano_inicial, 1, 1).strftime("%Y%m%d"))
    fim = int(datetime(ano_final, 12, 31).strftime("%Y%m%d"))

    base_url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        "?parameters=T2M_MIN,T2M_MAX,PRECTOTCORR,RH2M,WS2M,ALLSKY_SFC_SW_DWN,TOA_SW_DWN"
        "&community=RE&longitude={lon}&latitude={lat}&start={ini}&end={fim}&format=JSON"
    )
    resp = _com_timeout(
        requests.get, base_url.format(lon=longitude, lat=latitude, ini=ini, fim=fim), timeout=60
    )
    resp.raise_for_status()
    parametros = resp.json()["properties"]["parameter"]

    datas_str = sorted(parametros["T2M_MIN"].keys())
    datas = [pd.to_datetime(d) for d in datas_str]

    df = pd.DataFrame({"Data": datas})
    df["Tmin"] = [parametros["T2M_MIN"][d] for d in datas_str]
    df["Tmax"] = [parametros["T2M_MAX"][d] for d in datas_str]
    df["Chuva"] = [parametros["PRECTOTCORR"][d] for d in datas_str]
    df["UR"] = [parametros["RH2M"][d] for d in datas_str]
    df["U2"] = [parametros["WS2M"][d] for d in datas_str]
    df["Rs"] = np.array([parametros["ALLSKY_SFC_SW_DWN"][d] for d in datas_str]) * 3.6
    df["Qo"] = np.array([parametros["TOA_SW_DWN"][d] for d in datas_str]) * 3.6

    df["Tmed"] = (df["Tmax"] + df["Tmin"]) / 2
    df["Patm"] = 101.3 * ((293 - 0.0065 * altitude) / 293) ** 5.26
    df["NDA"] = df["Data"].dt.dayofyear

    df["d"] = df["NDA"].apply(_dec_sol)
    df["Hn"] = df["d"].apply(lambda d_: _hora_nascer_sol(d_, latitude))
    df["BOC"] = df["Rs"] * 0.75
    df["N"] = (2 * df["Hn"]) / 15
    df["n_inso"] = (((df["Rs"] / df["Qo"]) - (0.29 * np.cos(latitude))) / 0.52) * df["N"]

    df["es_Tmax"] = df["Tmax"].apply(_e_saturacao)
    df["es_Tmin"] = df["Tmin"].apply(_e_saturacao)
    df["es"] = (df["es_Tmax"] + df["es_Tmin"]) / 2
    df["ea"] = (df["UR"] / 100) * df["es"]
    df["QGcs"] = df["Qo"] * (0.75 + (2e-5) * altitude)

    df["BOL"] = _bol(df["Tmax"], df["Tmin"], df["ea"], df["Rs"], df["QGcs"])
    df["Rn"] = df["BOC"] + df["BOL"]
    df["s"] = (4098 * df["es"]) / (df["Tmed"] + 237.3) ** 2
    df["gama"] = 0.665e-3 * df["Patm"]
    df["ETo"] = _eto_penman(df["s"], df["Rn"], df["gama"], df["U2"], df["es"], df["ea"], df["Tmed"])

    df.replace(-999, np.nan, inplace=True)
    df.interpolate(inplace=True)
    return df


def simular_safra(clima: pd.DataFrame, dta: float, data_sem_str: str, ano_inicial: int) -> pd.DataFrame | None:
    """Roda o balanco hidrico por fase fenologica (F1-F4) para uma unica data de semeadura. Retorna None se o clima nao cobre o ciclo todo."""
    p = _parametros_milho()
    data_semeadura = datetime.strptime(f"{data_sem_str}{ano_inicial}", "%d/%m/%Y")
    data_maturacao = data_semeadura + timedelta(days=DURACAO_CICLO - 1)

    safra1 = clima.loc[(clima["Data"] >= data_semeadura) & (clima["Data"] <= data_maturacao)].copy()
    if len(safra1) < DURACAO_CICLO:
        return None
    safra1["n"] = range(1, len(safra1) + 1)

    fase_cultura, kc_lista = [], []
    for n in safra1["n"]:
        if n <= p["FASE1D"]:
            kc_lista.append(p["KC1"]); fase_cultura.append(1)
        elif n <= p["FASE1D"] + p["FASE2D"]:
            kc_lista.append(kc_lista[-1] + p["ACRESCIMO_KC1"]); fase_cultura.append(2)
        elif n <= p["FASE1D"] + p["FASE2D"] + p["FASE3D"]:
            kc_lista.append(p["KC2"]); fase_cultura.append(3)
        else:
            kc_lista.append(kc_lista[-1] + p["ACRESCIMO_KC2"]); fase_cultura.append(4)
    safra1["Kc"] = kc_lista
    safra1["Fase"] = fase_cultura

    z1 = []
    for n in safra1["n"]:
        if n == 1:
            z1.append(Z_INICIAL + p["ACRESCIMO_Z"])
        elif n <= p["FASE1D"] + p["FASE2D"]:
            z1.append(z1[-1] + p["ACRESCIMO_Z"])
        else:
            z1.append(p["Z_FINAL"])
    safra1["z"] = z1

    safra1["CAD"] = safra1["z"] * dta
    safra1["ETc"] = safra1["Kc"] * safra1["ETo"]
    safra1["P-ETc"] = safra1["Chuva"] - safra1["ETc"]

    petc = safra1["P-ETc"].to_numpy()
    cad = safra1["CAD"].to_numpy()
    arm = [Z_INICIAL * dta]
    for pp, c in zip(petc, cad):
        prev = arm[-1]
        if pp < 0:
            arm.append(prev * np.exp(pp / c))
        elif pp + prev >= c:
            arm.append(c)
        else:
            arm.append(prev + pp)
    arm = arm[1:]
    safra1["ARM"] = arm

    alt = [0] + list(np.array(arm[1:]) - np.array(arm[:-1]))
    safra1["ALT"] = alt

    safra1["ETR"] = np.where(safra1["P-ETc"] < 0, safra1["Chuva"] + np.abs(safra1["ALT"]), safra1["ETc"])
    safra1["DEF"] = safra1["ETc"] - safra1["ETR"]
    safra1["EXC"] = np.where(safra1["ARM"] < safra1["CAD"], 0, safra1["P-ETc"] - safra1["ALT"])
    safra1["ISNA_diario"] = safra1["ETR"] / safra1["ETc"]

    safra1["n/N"] = safra1["n_inso"] / safra1["N"]
    safra1["qo"] = safra1["Qo"] * 23.9234
    safra1["cTc"] = -4.16 + (0.4325 * safra1["Tmed"]) - (0.00725 * safra1["Tmed"] ** 2)
    safra1["cTn"] = -1.064 + (0.173 * safra1["Tmed"]) + (0.0029 * safra1["Tmed"] ** 2)
    safra1["PPBp"] = (
        (107.2 + 0.36 * safra1["qo"]) * safra1["cTc"] * safra1["n/N"]
        + (31.7 + 0.219 * safra1["qo"]) * safra1["cTn"] * (1 - safra1["n/N"])
    )
    safra1["CR"] = np.where(safra1["Tmed"] >= 20, 0.5, 0.6)
    safra1["PP"] = (safra1["PPBp"] * CIAFMAX * safra1["CR"] * CC) / (1 - 0.01 * U)

    bp = safra1["PP"].sum()

    def rc_fase(a, b):
        return safra1["ETR"].iloc[a:b].sum() / safra1["ETc"].iloc[a:b].sum()

    rc_1 = rc_fase(0, int(p["FASE1D"]))
    rc_2 = rc_fase(int(p["FASE1D"]), int(p["FASE1D"] + p["FASE2D"]))
    rc_3 = rc_fase(int(p["FASE1D"] + p["FASE2D"]), int(p["FASE1D"] + p["FASE2D"] + p["FASE3D"]))
    rc_4 = rc_fase(int(p["FASE1D"] + p["FASE2D"] + p["FASE3D"]), int(p["FASE1D"] + p["FASE2D"] + p["FASE3D"] + p["FASE4D"]))

    pa1 = (1 - p["KY1"] * (1 - rc_1)) * bp
    pa2 = (1 - p["KY2"] * (1 - rc_2)) * pa1
    pa3 = (1 - p["KY3"] * (1 - rc_3)) * pa2
    pa4 = (1 - p["KY4"] * (1 - rc_4)) * pa3

    safra = safra1.groupby("Fase").agg(
        Tmin=("Tmin", "mean"), Tmax=("Tmax", "mean"), Tmed=("Tmed", "mean"),
        Chuva=("Chuva", "sum"), UR=("UR", "mean"), ETc=("ETc", "sum"), ETR=("ETR", "sum"),
        ARM=("ARM", "last"), DEF=("DEF", "sum"), EXC=("EXC", "sum"), N=("N", "mean"), PP=("PP", "sum"),
    ).reset_index()
    safra["ISNA"] = [rc_1, rc_2, rc_3, rc_4]
    safra["PA"] = [pa1, pa2, pa3, pa4]
    safra["Data semeadura"] = data_semeadura
    safra["Data maturação"] = data_maturacao
    return safra


def _construir_features(safra: pd.DataFrame, municipio: dict, ano: int) -> pd.DataFrame:
    df = safra.copy()
    df["Municipio"] = municipio["municipio"]
    df["Ano"] = ano
    df["Latitude"] = municipio["latitude"]
    df["Longitude"] = municipio["longitude"]
    df["Altitude"] = municipio["altitude"]

    df["Amp"] = df["Tmax"] - df["Tmin"]
    df["Pefetiva"] = df["Chuva"] / df["ETc"]
    df["T_DEF"] = df["Tmax"] * df["DEF"]

    df["Data semeadura"] = pd.to_datetime(df["Data semeadura"])
    df["Data maturação"] = pd.to_datetime(df["Data maturação"])
    df["DOY_semeadura"] = df["Data semeadura"].dt.dayofyear
    df["DOY_maturacao"] = df["Data maturação"].dt.dayofyear
    df["lat_lon"] = df["Latitude"] * df["Longitude"]
    df = df.drop(columns=["Data semeadura", "Data maturação"])

    df["rep"] = 0
    largo = df.pivot(
        index=["Municipio", "Ano", "DOY_semeadura", "DOY_maturacao", "rep"], columns="Fase", values=VARS_FASE
    )
    largo.columns = [f"{v}_F{f}" for v, f in largo.columns]
    largo = largo.reset_index()
    largo = largo.drop(
        columns=[
            c for c in largo.columns
            if c.split("_F")[-1] in ("2", "3", "4") and c.split("_F")[0] in ("Altitude", "Latitude", "Longitude", "lat_lon")
        ],
        errors="ignore",
    )
    return largo


@lru_cache(maxsize=1)
def _carregar_modelo():
    caminho = CACHE_DIR / "ExtraTrees.joblib"
    if not caminho.exists():
        resp = _com_timeout(requests.get, URL_MODELO_EXTRATREES, timeout=120)
        resp.raise_for_status()
        caminho.write_bytes(resp.content)
    # joblib.load tambem passa por _com_timeout: leitura de um arquivo de 55MB pode ser
    # atrasada por antivirus/EDR escaneando o arquivo na primeira leitura por um processo
    # novo (observado neste projeto — nao acontece rodando python direto no terminal, so
    # quando o processo e filho do Claude Code/VS Code).
    return _com_timeout(joblib.load, caminho, timeout=90)


def prever_produtividade(municipio: str, ano: int, data_semeadura: str = "15/03") -> dict:
    """Roda o pipeline completo (clima NASA POWER -> balanco hidrico FAO -> features -> ExtraTrees.predict) para 1 municipio/ano/data.

    `data_semeadura` deve ser uma das 6 datas em que o modelo foi treinado: 05/02, 15/02, 25/02, 05/03, 15/03, 25/03.
    """
    if data_semeadura not in DATAS_SEMEADURA_SUPORTADAS:
        raise ValueError(
            f"data_semeadura '{data_semeadura}' nao suportada. Use uma de: {DATAS_SEMEADURA_SUPORTADAS} "
            "(o modelo so foi treinado nessas 6 datas)."
        )

    muni = buscar_municipio(municipio)
    clima = buscar_clima_diario(muni["latitude"], muni["longitude"], muni["altitude"], ano)
    safra = simular_safra(clima, muni["dta"], f"{data_semeadura}/", ano)
    if safra is None:
        raise ValueError(
            f"Sem dados climaticos suficientes para cobrir o ciclo completo (100 dias) a partir de {data_semeadura}/{ano} em {muni['municipio']}."
        )

    largo = _construir_features(safra, muni, ano)
    faltando = [f for f in FEATURES if f not in largo.columns]
    if faltando:
        raise RuntimeError(f"Features ausentes apos o pivot: {faltando}")

    modelo = _carregar_modelo()
    produtividade = float(modelo.predict(largo[FEATURES])[0])

    return {
        "municipio": muni["municipio"],
        "estado": muni["estado"],
        "ano": ano,
        "data_semeadura": data_semeadura,
        "latitude": muni["latitude"],
        "longitude": muni["longitude"],
        "altitude": muni["altitude"],
        "produtividade_prevista_kg_ha": produtividade,
        "isna_por_fase": largo[[c for c in largo.columns if c.startswith("ISNA_F")]].iloc[0].to_dict(),
    }
