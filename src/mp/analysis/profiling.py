"""Perfil do dataset: o que veio, quanto veio e em que ritmo.

Descreve. Nao corrige.
"""

from __future__ import annotations

import re

import pandas as pd

from mp import config, segmentos


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

    Base do **G2**, o guardrail que encerra o fluxo prescritivo quando nao ha
    defeito a corrigir. Casa por radical porque os dados trazem variacoes
    (`normal_2`, `new_normal_6`) e typos (`normla_carga_3_3`).

    **Limite conhecido:** substring nao pega rotulo truncado — `new_tes` nao
    contem `teste` e cai como defeito. Por isso a decisao que vale e tomada no
    nivel da familia, em `sugerir_familias`, que agrupa pelo prefixo.
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

    Acrescenta `sessao` — blocos de coleta separados por mais de
    `GAP_NOVA_SESSAO_S` — e `delta_s`, o intervalo desde a leitura anterior
    dentro da sessao.

    A fronteira de sessao existe porque o arquivo mistura campanhas gravadas com
    semanas de distancia: sem ela, um grafico de linha ligaria o fim de uma ao
    inicio de outra e inventaria uma transicao.
    """
    tempo = config.COLUNA_TEMPO
    sub = df if rotulo is None else df[df[config.COLUNA_ROTULO] == rotulo]
    sub = sub.sort_values(tempo, kind="stable").reset_index(drop=True)

    if sub.empty:
        return sub.assign(sessao=pd.Series(dtype=int), delta_s=pd.Series(dtype=float))

    gap = sub[tempo].diff().dt.total_seconds()
    corte = segmentos.passou_intervalo(sub[tempo], config.GAP_NOVA_SESSAO_S)
    # `- 1` porque `numerar_grupos` comeca em 1 e a sessao e contada a partir de 0.
    sub["sessao"] = segmentos.numerar_grupos(corte) - 1
    sub["delta_s"] = gap.where(gap <= config.GAP_NOVA_SESSAO_S)
    return sub


def analise_intervalos(df: pd.DataFrame, cortes=None) -> dict:
    """Justifica numericamente onde cortar um episodio.

    Um episodio termina quando a coleta para, mas "parar" precisa de um numero
    de segundos, e no chute ele nao se defende. So contam os intervalos dentro
    do mesmo rotulo: na troca de rotulo o episodio quebra de qualquer jeito.

    O argumento central esta em `vazio` — a faixa sem nenhuma observacao entre a
    cadencia normal e as paradas reais. Qualquer corte ali dentro da o mesmo
    resultado, entao a escolha deixa de ser arbitraria.

    Traz ainda `estatisticas`, `faixas`, `paradas` (so as interrupcoes de
    verdade) e `sensibilidade`, quantos episodios cada corte produziria.
    """
    tempo, rot = config.COLUNA_TEMPO, config.COLUNA_ROTULO

    ordenado = df.sort_values(tempo, kind="stable")
    gap = ordenado[tempo].diff().dt.total_seconds()
    mesmo_rotulo = (ordenado[rot] == ordenado[rot].shift()).fillna(False).to_numpy(bool)

    dentro = gap.to_numpy()[mesmo_rotulo]
    dentro = dentro[~pd.isna(dentro)]

    if dentro.size == 0:
        return {"estatisticas": {}, "faixas": pd.DataFrame(), "vazio": {},
                "paradas": {}, "sensibilidade": pd.DataFrame(), "intervalos": dentro}

    estatisticas = {
        "n": int(dentro.size),
        "minimo_s": float(dentro.min()),
        "mediana_s": float(pd.Series(dentro).median()),
        "media_s": float(dentro.mean()),
        "maximo_s": float(dentro.max()),
    }

    limites = [0, 1, 2.5, 4, 6, 10, 15, 20, 30, 60, 300, 3600, float("inf")]
    faixas = pd.DataFrame(
        {
            "de_s": limites[:-1],
            "ate_s": limites[1:],
            "intervalos": [
                int(((dentro >= a) & (dentro < b)).sum())
                for a, b in zip(limites[:-1], limites[1:])
            ],
        }
    )
    faixas["vazia"] = faixas["intervalos"] == 0

    # A cadencia nominal e ~2 s; toleramos ate 10 s para nao cortar o segundo
    # modo de 5,3 s observado no dataset. Acima disso e parada.
    LIMITE_CADENCIA = 10.0
    cadencia = dentro[dentro <= LIMITE_CADENCIA]
    paradas = dentro[dentro > LIMITE_CADENCIA]

    vazio = {
        "maior_cadencia_s": float(cadencia.max()) if cadencia.size else None,
        "menor_parada_s": float(paradas.min()) if paradas.size else None,
    }
    if vazio["maior_cadencia_s"] is not None and vazio["menor_parada_s"] is not None:
        vazio["largura_s"] = vazio["menor_parada_s"] - vazio["maior_cadencia_s"]
        vazio["centro_s"] = (vazio["menor_parada_s"] + vazio["maior_cadencia_s"]) / 2

    resumo_paradas = (
        {
            "n": int(paradas.size),
            "minima_s": float(paradas.min()),
            "mediana_s": float(pd.Series(paradas).median()),
            "media_s": float(paradas.mean()),
            "maxima_s": float(paradas.max()),
        }
        if paradas.size
        else {}
    )

    # --- sensibilidade -----------------------------------------------------
    # Conta episodios para varios cortes. Nao e a implementacao da Parte 1 —
    # aqui so precisamos do NUMERO, para mostrar onde ele fica estavel.
    cortes = list(cortes) if cortes else [2.5, 5, 8, 10, 12, 15, 20, 30, 45, 60, 120, 300]
    muda_rotulo = segmentos.mudou_valor(ordenado[rot])

    sensibilidade = pd.DataFrame(
        {
            "corte_s": cortes,
            "episodios": [
                int(
                    segmentos.numerar_grupos(
                        muda_rotulo, segmentos.passou_intervalo(ordenado[tempo], c)
                    )[-1]
                )
                for c in cortes
            ],
        }
    )

    return {
        "estatisticas": estatisticas,
        "faixas": faixas,
        "vazio": vazio,
        "paradas": resumo_paradas,
        "sensibilidade": sensibilidade,
        "intervalos": dentro,
    }


def serie_temporal(
    df: pd.DataFrame,
    rotulo: str,
    colunas: list[str],
    max_pontos: int | None = None,
) -> dict:
    """Serie temporal de um rotulo, pronta para plotar.

    Formato longo, que e o que o Altair consome direto com `coluna` virando
    facet.

    Um rotulo pode ter 17 mil leituras e travar o navegador, entao acima de
    `max_pontos` os pontos viram blocos com mediana, minimo e maximo. A mediana
    da a tendencia; a faixa min-max preserva os picos, que e o que interessa em
    vibracao — so a media apagaria o impacto isolado de defeito de rolamento.

    Blocos por POSICAO, nao por janela de tempo: as sessoes estao separadas por
    ate 122 h e uma janela fixa produziria milhares de blocos vazios.
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


def serie_bruta(
    df: pd.DataFrame,
    colunas: list[str],
    inicio: int = 0,
    quantidade: int = 2000,
    rotulos: list[str] | None = None,
) -> dict:
    """Um trecho do arquivo **na ordem em que foi lido**, sem nenhum tratamento.

    O eixo x e a posicao da linha no arquivo, nao a data; nada e agregado,
    deduplicado nem reordenado. Existe para tornar visivel o que as outras telas
    corrigem em silencio: o arquivo nao esta em ordem de data, entao avancar uma
    linha pode significar voltar semanas no tempo.

    `rotulos` escolhe quais linhas aparecem e nada mais. A posicao fisica e
    capturada antes do filtro, entao a leitura da linha 90.000 continua desenhada
    em 90.000 mesmo sendo a terceira do recorte — senao o grafico mentiria sobre
    onde o trecho esta. Os buracos que o filtro abre viram quebra de `bloco`,
    para a linha se interromper em vez de atravessar o vao.

    A janela e limitada porque 166 mil pontos sem reamostrar nao cabem no
    navegador. Devolve tambem `faixas`, os blocos de mesmo `fault`, e `trocas`,
    as linhas onde o rotulo mudou — salto negativo = a seguinte e mais antiga.
    """
    tempo, rot = config.COLUNA_TEMPO, config.COLUNA_ROTULO

    total_arquivo = len(df)

    # A posicao fisica no arquivo, guardada ANTES do filtro — e ela que vira o
    # eixo x. `iloc` e nao `loc`: o indice do DataFrame pode nao coincidir com
    # a posicao.
    base = df.copy()
    base["_posicao"] = range(total_arquivo)

    selecionados = [r for r in (rotulos or []) if r]
    if selecionados:
        base = base[base[rot].isin(selecionados)]

    # O navegador de trecho anda dentro do que sobrou do filtro.
    total = len(base)
    inicio = max(0, min(int(inicio), max(0, total - 1)))
    fim = min(total, inicio + int(quantidade))

    janela = base.iloc[inicio:fim].copy()
    janela["linha"] = janela["_posicao"].to_numpy()

    colunas = [c for c in colunas if c in janela.columns]

    vazio = pd.DataFrame(columns=["linha", tempo, rot, "coluna", "valor"])
    if janela.empty or not colunas:
        return {"dados": vazio, "faixas": pd.DataFrame(), "trocas": pd.DataFrame(),
                "inicio": inicio, "fim": fim, "total": total,
                "total_arquivo": total_arquivo, "n_linhas": 0, "rotulos": [],
                "filtrado": bool(selecionados), "rotulos_filtro": selecionados,
                "linha_inicio": 0, "linha_fim": 0, "n_blocos": 0}

    # --- em que ponto a serie se parte -------------------------------------
    #
    # Duas coisas distintas, de proposito separadas:
    #
    # `muda_rotulo`  o valor de `fault` mudou de uma linha para a outra. E o que
    #                a tabela de trocas relata, e so isso conta como troca.
    # `salta_linha`  as duas linhas nao eram vizinhas no arquivo. So acontece
    #                com filtro ligado, e nao e troca nenhuma — e um buraco
    #                aberto pela selecao.
    #
    # O bloco quebra nos dois casos; a troca, so no primeiro. Misturar os dois
    # faria o filtro inventar trocas de rotulo que o arquivo nao tem.
    muda_rotulo = segmentos.mudou_valor(janela[rot])
    salta_linha = janela["linha"].diff().ne(1).to_numpy(dtype=bool)
    muda_bloco = muda_rotulo | salta_linha

    grupo = segmentos.numerar_grupos(muda_bloco)
    janela["bloco"] = grupo

    dados = janela.melt(
        id_vars=["linha", tempo, rot, "bloco"], value_vars=colunas,
        var_name="coluna", value_name="valor",
    )

    # `valor_z` existe so para poder desenhar tudo num grafico so. As unidades
    # nao sao comparaveis — rpm anda na casa dos milhares e kurtosis fica perto
    # de 2,5; no mesmo eixo, a segunda vira uma reta colada no zero.
    #
    # A referencia e a coluna INTEIRA, nao a janela: assim o valor lido continua
    # dizendo onde aquele trecho esta em relacao a todo o arquivo. Padronizar
    # dentro da janela faria um trecho totalmente normal parecer cheio de picos,
    # porque a escala se ajustaria ao proprio ruido.
    referencia = df[colunas].agg(["mean", "std"])
    media = dados["coluna"].map(referencia.loc["mean"]).astype(float)
    desvio = dados["coluna"].map(referencia.loc["std"]).astype(float)
    # Coluna constante tem desvio zero: fica em zero em vez de virar infinito.
    dados["valor_z"] = ((dados["valor"] - media) / desvio.where(desvio > 0)).fillna(0.0)

    # --- onde o rotulo troca ------------------------------------------------
    faixas = (
        janela.assign(_g=grupo)
        .groupby("_g", sort=True)
        .agg(
            **{
                "linha_inicio": ("linha", "min"),
                "linha_fim": ("linha", "max"),
                rot: (rot, "first"),
                "inicio": (tempo, "min"),
                "fim": (tempo, "max"),
                "n_linhas": ("linha", "size"),
            }
        )
        .reset_index(drop=True)
    )
    # O retangulo vai ate o inicio do bloco seguinte, senao ficam buracos de uma
    # linha entre as faixas. O ultimo fecha na ultima linha DESENHADA — que com
    # filtro nao e `fim`, porque `fim` conta o recorte e o eixo conta o arquivo.
    ultima_linha = int(janela["linha"].max()) + 1
    faixas["ate"] = faixas["linha_inicio"].shift(-1).fillna(ultima_linha).astype(int)
    # Meio do bloco: onde o nome do rotulo e escrito, para ficar sempre legivel
    # sem depender do mouse.
    faixas["centro"] = (faixas["linha_inicio"] + faixas["ate"]) / 2

    salto = janela[tempo].diff().dt.total_seconds()
    trocas = pd.DataFrame(
        {
            "linha": janela["linha"].to_numpy(),
            "de": janela[rot].shift().to_numpy(),
            "para": janela[rot].to_numpy(),
            tempo: janela[tempo].to_numpy(),
            "salto_s": salto.to_numpy(),
        }
    )
    # A primeira linha da janela nao e uma troca — e so o comeco do recorte.
    # Comparar com `inicio` nao serve mais: com filtro, `inicio` e posicao no
    # recorte e `linha` e posicao no arquivo. A primeira e a primeira, e ponto.
    nao_e_a_primeira = pd.Series(True, index=range(len(janela)))
    nao_e_a_primeira.iloc[0] = False

    # `~salta_linha`: so e troca se as duas leituras eram MESMO vizinhas no
    # arquivo. Com filtro, o rotulo tambem "muda" ao pular de uma linha para
    # outra a 35 mil de distancia — mas isso e o buraco do filtro, nao uma
    # troca. Sem esta condicao a tabela de trocas relataria saltos de tempo
    # gigantes como se fossem emenda entre gravacoes, que e outra coisa.
    trocas = trocas[
        muda_rotulo & ~salta_linha & nao_e_a_primeira.to_numpy()
    ].reset_index(drop=True)

    # Etiqueta pronta para desenhar no grafico, sempre visivel — sem depender de
    # passar o mouse.
    trocas["etiqueta"] = [
        f"{'?' if pd.isna(de) else de} → {para}"
        for de, para in zip(trocas["de"], trocas["para"])
    ]
    # Alterna a altura das etiquetas para duas trocas proximas nao se cobrirem.
    trocas["fila"] = [i % 2 for i in range(len(trocas))]

    return {
        "dados": dados,
        "faixas": faixas,
        "trocas": trocas,
        "inicio": inicio,
        "fim": fim,
        "total": total,
        "total_arquivo": total_arquivo,
        "n_linhas": len(janela),
        "rotulos": [str(r) for r in faixas[rot].tolist()],
        "bruto": janela,
        # Contexto do filtro, para a tela poder avisar o que esta escondendo.
        "filtrado": bool(selecionados),
        "rotulos_filtro": selecionados,
        # Extremos REAIS no arquivo — com filtro, o recorte pode ir da linha 12
        # a 90.000 e mostrar so algumas centenas no meio.
        "linha_inicio": int(janela["linha"].min()),
        "linha_fim": int(janela["linha"].max()),
        "n_blocos": int(pd.Series(grupo).nunique()),
    }


def saltos_no_arquivo(df: pd.DataFrame) -> dict:
    """Quanto o tempo anda para tras quando se le o arquivo linha a linha.

    O numero que justifica a tela de dados brutos. Nas outras paginas o dataset
    ja chega ordenado; aqui medimos o estrago de nao ordenar.
    """
    tempo = config.COLUNA_TEMPO
    delta = df[tempo].diff().dt.total_seconds().dropna()
    para_tras = delta[delta < 0]

    return {
        "linhas": len(df),
        "em_ordem": bool(df[tempo].is_monotonic_increasing),
        "saltos_para_tras": int(len(para_tras)),
        "pct_para_tras": round(float(len(para_tras) / len(delta) * 100), 3) if len(delta) else 0.0,
        "maior_recuo_dias": round(float(-para_tras.min()) / 86400, 2) if len(para_tras) else 0.0,
        "avanco_mediano_s": round(float(delta[delta >= 0].median()), 3) if (delta >= 0).any() else 0.0,
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
