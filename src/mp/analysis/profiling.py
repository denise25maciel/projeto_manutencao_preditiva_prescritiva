"""Perfil do dataset: o que veio, quanto veio e em que ritmo.

Descreve. Nao corrige.
"""

from __future__ import annotations

import re

import pandas as pd

from mp import config


def resumo_geral(df: pd.DataFrame) -> dict:
    """Numeros de cabecalho para o topo do dashboard."""
    return {
        "linhas": len(df),
        "colunas": df.shape[1],
        "rotulos_distintos": int(df[config.COLUNA_ROTULO].nunique()),
        "celulas_nulas": int(df.isna().sum().sum()),
        "memoria_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 1),
    }


def nulos_por_coluna(df: pd.DataFrame) -> pd.DataFrame:
    """Contagem e percentual de nulos por coluna, mais o tipo lido.

    Cobre so o nulo declarado (NaN / campo vazio). Valor sentinela — um 0.0 que
    na verdade significa "sensor sem leitura" — nao aparece aqui; quem levanta
    essa suspeita e a analise de outliers e a de constantes.
    """
    n = len(df)
    fora = df.isna().sum()

    tabela = pd.DataFrame(
        {
            "coluna": fora.index,
            "tipo": [str(df[c].dtype) for c in fora.index],
            "nulos": fora.to_numpy(),
            "pct_nulos": (fora.to_numpy() / n * 100).round(3) if n else 0.0,
            "preenchidos": n - fora.to_numpy(),
            "distintos": [int(df[c].nunique(dropna=True)) for c in fora.index],
        }
    )
    return tabela.sort_values(["nulos", "coluna"], ascending=[False, True]).reset_index(drop=True)


def perfil_rotulos(df: pd.DataFrame) -> pd.DataFrame:
    """Um registro por valor distinto de `fault`.

    Alem da contagem, traz a janela temporal de cada rotulo. Rotulo cujo
    primeiro e ultimo registro estao a poucos minutos de distancia foi coletado
    numa sessao unica — contar suas linhas como "ocorrencias" seria enganoso.
    Esse e o argumento para os episodios da Parte 1.
    """
    rot, tempo = config.COLUNA_ROTULO, config.COLUNA_TEMPO
    n = len(df)

    linhas = df.groupby(rot, dropna=False, observed=True).agg(n_leituras=(rot, "size"))

    if tempo in df.columns:
        janelas = df.groupby(rot, dropna=False, observed=True)[tempo].agg(["min", "max"])
        linhas = linhas.join(janelas).rename(columns={"min": "primeira", "max": "ultima"})
        # Duracao coberta pelo rotulo. Nao e tempo de operacao: se o rotulo
        # aparece em duas sessoes distantes, o span engloba o intervalo morto.
        linhas["span_horas"] = (
            (linhas["ultima"] - linhas["primeira"]).dt.total_seconds() / 3600
        ).round(2)

    linhas["pct"] = (linhas["n_leituras"] / n * 100).round(3)
    linhas["e_problema"] = [not e_estado(str(r)) for r in linhas.index]

    return linhas.reset_index().sort_values("n_leituras", ascending=False).reset_index(drop=True)


def e_estado(rotulo: str) -> bool:
    """True se o rotulo descreve um ESTADO da maquina, nao um defeito.

    Base do guardrail G2. Casa por radical porque os dados trazem variacoes
    (`normal_2`, `normal_carga_3_3`, `new_normal_6`) e typos (`normla_carga_3_3`).

    **Limite conhecido:** casar por substring nao pega rotulo TRUNCADO. `new_tes`
    (2 leituras) e uma abreviacao de `new_teste`, mas nao contem o radical
    `teste` e por isso e classificado como defeito aqui. Por isso a decisao que
    vale para os guardrails e tomada no nivel da **familia**, nao do rotulo
    solto — ver `sugerir_familias`, que agrupa `new_tes` em `teste` pelo prefixo.
    """
    r = rotulo.lower()
    return any(radical in r for radical in config.RADICAIS_NAO_PROBLEMA)


def janela_temporal(df: pd.DataFrame) -> dict:
    """Inicio, fim e continuidade da coleta.

    `monotonico` responde se o arquivo esta em ordem cronologica. No banner.csv
    ele NAO esta: ha saltos negativos de dezenas de dias, sinal de que sessoes
    gravadas em epocas diferentes foram concatenadas fora de ordem. Qualquer
    janela deslizante (a mediana movel da Parte 3) precisa ordenar antes.
    """
    t = df[config.COLUNA_TEMPO]
    return {
        "inicio": t.min(),
        "fim": t.max(),
        "duracao_dias": round((t.max() - t.min()).total_seconds() / 86400, 2),
        "monotonico": bool(t.is_monotonic_increasing),
        "timestamps_distintos": int(t.nunique()),
        "timestamps_repetidos": int(len(t) - t.nunique()),
    }


def taxa_amostragem(df: pd.DataFrame) -> dict:
    """Distribuicao do intervalo entre leituras consecutivas.

    Ordenamos por tempo antes de derivar, senao a desordem do arquivo produz
    intervalos negativos que nao existem na coleta real.

    `pct_na_cadencia` mede quanto da coleta respeita os ~2s esperados.
    `cortes` conta os intervalos acima de GAP_NOVA_SESSAO_S: sao fronteiras
    entre sessoes, nao falhas de amostragem.
    """
    t = df[config.COLUNA_TEMPO].sort_values()
    d = t.diff().dt.total_seconds().dropna()

    alvo, tol = config.INTERVALO_ESPERADO_S, config.TOLERANCIA_INTERVALO_S
    na_cadencia = d.between(alvo - tol, alvo + tol)
    cortes = d > config.GAP_NOVA_SESSAO_S

    return {
        "intervalo_mediano_s": round(float(d.median()), 3),
        "intervalo_p05_s": round(float(d.quantile(0.05)), 3),
        "intervalo_p95_s": round(float(d.quantile(0.95)), 3),
        "pct_na_cadencia": round(float(na_cadencia.mean() * 100), 2),
        "cortes": int(cortes.sum()),
        "sessoes_estimadas": int(cortes.sum()) + 1,
        "maior_gap_horas": round(float(d.max()) / 3600, 2),
        # Serie completa para o histograma da UI, ja limitada a cadencia curta
        # (o eixo ficaria ilegivel com um gap de 30 dias no grafico).
        "intervalos": d[d <= 30].to_numpy(),
    }


def ordenar_por_tempo(df: pd.DataFrame, rotulo: str | None = None) -> pd.DataFrame:
    """Recorta um rotulo e devolve suas leituras em ordem cronologica.

    Acrescenta duas colunas derivadas:

    - `sessao` — sessoes de coleta separadas por um intervalo maior que
      `GAP_NOVA_SESSAO_S`. O arquivo nao esta em ordem e mistura campanhas
      gravadas com semanas de distancia; sem marcar a fronteira, qualquer
      grafico de linha ligaria o fim de uma sessao ao inicio de outra e
      inventaria uma transicao que nunca existiu.
    - `delta_s` — intervalo desde a leitura anterior DENTRO da sessao.
    """
    tempo = config.COLUNA_TEMPO
    sub = df if rotulo is None else df[df[config.COLUNA_ROTULO] == rotulo]
    sub = sub.sort_values(tempo, kind="stable").reset_index(drop=True)

    if sub.empty:
        return sub.assign(sessao=pd.Series(dtype=int), delta_s=pd.Series(dtype=float))

    gap = sub[tempo].diff().dt.total_seconds()
    # O primeiro gap e NaN; `NaN > x` e False, entao a primeira linha cai na
    # sessao 0 sem tratamento especial.
    sub["sessao"] = (gap > config.GAP_NOVA_SESSAO_S).cumsum().astype(int)
    sub["delta_s"] = gap.where(gap <= config.GAP_NOVA_SESSAO_S)
    return sub


def serie_temporal(
    df: pd.DataFrame,
    rotulo: str,
    colunas: list[str],
    max_pontos: int | None = None,
) -> dict:
    """Serie temporal de um rotulo, pronta para plotar.

    Devolve formato longo — `[created_at, sessao, coluna, valor, minimo, maximo]` —
    porque e o que o Altair consome direto, com `coluna` virando facet.

    **Reamostragem.** Um rotulo pode ter 17 mil leituras; mandar tudo para o
    navegador trava a pagina. Acima de `max_pontos` agrupamos em blocos
    consecutivos DENTRO de cada sessao e reportamos mediana, minimo e maximo do
    bloco. A mediana da a tendencia; a faixa min-max preserva os picos, que em
    vibracao sao justamente o que interessa — reamostrar so pela media apagaria
    o impacto isolado que caracteriza defeito de rolamento.

    Os blocos sao por POSICAO, nao por janela de tempo: as sessoes estao
    separadas por ate 122 h, e uma janela temporal fixa produziria milhares de
    blocos vazios entre elas.
    """
    max_pontos = max_pontos or config.MAX_PONTOS_SERIE
    tempo = config.COLUNA_TEMPO

    sub = ordenar_por_tempo(df, rotulo)
    colunas = [c for c in colunas if c in sub.columns]
    if sub.empty or not colunas:
        vazio = pd.DataFrame(
            columns=[tempo, "sessao", "coluna", "valor", "minimo", "maximo"]
        )
        return {"dados": vazio, "n_original": 0, "n_pontos": 0,
                "reamostrado": False, "fator": 1, "sessoes": 0, "ordenado": sub}

    n = len(sub)
    fator = max(1, -(-n // max_pontos))  # ceil

    if fator > 1:
        # cumcount por sessao garante que um bloco nunca cruze a fronteira.
        sub = sub.assign(_bloco=sub.groupby("sessao").cumcount() // fator)
        chaves = ["sessao", "_bloco"]
    else:
        sub = sub.assign(_bloco=range(n))
        chaves = ["sessao", "_bloco"]

    partes = []
    for c in colunas:
        g = sub.groupby(chaves, sort=True)
        bloco = g.agg(
            **{
                tempo: (tempo, "first"),
                "valor": (c, "median"),
                "minimo": (c, "min"),
                "maximo": (c, "max"),
            }
        ).reset_index()
        bloco["coluna"] = c
        partes.append(bloco[[tempo, "sessao", "coluna", "valor", "minimo", "maximo"]])

    dados = pd.concat(partes, ignore_index=True).sort_values([ "coluna", tempo])

    return {
        "dados": dados.reset_index(drop=True),
        "n_original": n,
        "n_pontos": int(len(dados) / len(colunas)),
        "reamostrado": fator > 1,
        "fator": fator,
        "sessoes": int(sub["sessao"].nunique()),
        "ordenado": sub.drop(columns="_bloco"),
    }


# --------------------------------------------------------------------------
# Agrupamento sugerido de rotulos
# --------------------------------------------------------------------------

# Familias-alvo, na ordem em que sao testadas. A primeira que casar vence, por
# isso as mais especificas vem antes (rolamento_inner antes de rolamento).
_FAMILIAS = [
    ("rolamento_inner", r"rolamento_inner|rolamento.*inner"),
    ("rolamento_outer", r"rolamento_outer|rolamento.*outer"),
    ("rolamento_ball", r"rolamento_ball|rolamento.*ball"),
    ("rolamento_combination", r"rolamento_comb"),
    ("desalinhamento", r"desalinhad|desalinham"),
    # Typos observados: desabalanceado, desbanlanceado, ddesbalanceado,
    # dedesbalanceado, desabanceado. O regex cobre as variacoes de metatese.
    ("desbalanceamento", r"d+e+s+a?b+a?n?l?a?n?c+e+a?d?|desbalanceamento"),
    ("cocked_rotor", r"cocked|cockecocked"),
    ("eccentric_rotor", r"eccentric"),
    ("polia", r"polia"),
    ("correia", r"correia"),
    ("ventoinha", r"ventoinha"),
    ("falta_fase", r"falta_fase"),
    ("motor_desligado", r"m[oa]r?tor_desligado"),
    ("normal", r"normal|normla|baseline"),
    ("teste", r"^new_tes|^teste$|_teste$"),
    ("acelerando", r"acelerando"),
]


def sugerir_familias(rotulos) -> pd.DataFrame:
    """Propoe uma familia para cada rotulo cru.

    SUGESTAO, nao decisao. O `fault_map.yaml` da Parte 1 e curado a mao e e
    ele que vale — este heuristico existe so para que a lista de 151 rotulos
    seja navegavel e para expor quais nao casam com nenhuma familia conhecida
    (coluna familia = "?", que e exatamente onde o curador deve olhar).
    """
    linhas = []
    for r in sorted({str(x) for x in rotulos}):
        base = r.lower()
        familia = "?"
        for nome, padrao in _FAMILIAS:
            if re.search(padrao, base):
                familia = nome
                break
        linhas.append(
            {
                "fault": r,
                "familia_sugerida": familia,
                "e_problema": not e_estado(r),
                "tem_prefixo_sessao": base.startswith(config.PREFIXOS_DE_SESSAO),
                "tem_sufixo_sessao": any(s in base for s in config.SUFIXOS_DE_SESSAO),
            }
        )
    return pd.DataFrame(linhas)
