"""Transformacao das leituras de sensor em eventos.

Divisao de responsabilidade dentro do projeto:

- `analysis/`  — **descreve** o dado bruto. Nao altera nada.
- `ingestion/` — **transforma** o dado bruto no que vai para o banco.
- `segmentos`  — a primitiva de agrupar linhas consecutivas, usada pelos dois.

Este arquivo e o lado do sensor; `documents.py` e o lado dos PDFs.

O que e um evento
-----------------
Uma vez em que a maquina foi medida com o mesmo defeito, sem interrupcao.

Contar linhas engana: `rolamento_inner` tem 13 mil linhas, mas foram 12
medicoes de meia hora. A pergunta "quantas vezes isso aconteceu?" so tem
resposta depois de agrupar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mp import config, segmentos

# Regra atual: um evento novo comeca quando o rotulo muda. Nada mais.
#
# `limite_intervalo_s` existe e vem desligado. A analise da tela "Qualidade dos
# Dados" mostra que um corte de 10 s separaria ensaios que hoje ficam juntos —
# mas a decisao foi comecar so pelo rotulo. O parametro fica pronto para quando
# a decisao mudar, e `diagnostico_eventos` reporta o custo de mante-lo desligado.
LIMITE_INTERVALO_PADRAO: float | None = None


def construir_eventos(
    df: pd.DataFrame,
    limite_intervalo_s: float | None = LIMITE_INTERVALO_PADRAO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agrupa as leituras em eventos.

    Devolve `(leituras, eventos)`:

    - `leituras` — o DataFrame de entrada ordenado por data, com a coluna
      `evento` dizendo a que evento cada linha pertence. Nenhuma linha e
      removida ou alterada.
    - `eventos`  — uma linha por evento: rotulo, quantas leituras, inicio, fim
      e duracao.

    A ordenacao por data e obrigatoria e feita aqui: o arquivo bruto vem fora de
    ordem, e agrupar linhas vizinhas num arquivo desordenado nao significa nada.
    """
    tempo, rotulo = config.COLUNA_TEMPO, config.COLUNA_ROTULO

    leituras = df.sort_values(tempo, kind="stable").reset_index(drop=True)
    if leituras.empty:
        vazio = pd.DataFrame(
            columns=["evento", rotulo, "n_leituras", "inicio", "fim",
                     "duracao_s", "duracao_min"]
        )
        return leituras.assign(evento=pd.Series(dtype="int64")), vazio

    cortes = [segmentos.mudou_valor(leituras[rotulo])]
    if limite_intervalo_s is not None:
        cortes.append(segmentos.passou_intervalo(leituras[tempo], limite_intervalo_s))

    grupos = segmentos.numerar_grupos(*cortes)
    leituras["evento"] = grupos

    eventos = segmentos.resumir_grupos(
        leituras, grupos, coluna_tempo=tempo, primeiro_de=[rotulo]
    ).rename(columns={"_grupo": "evento"})

    return leituras, eventos


def construir_eventos_por_rotulo(
    df: pd.DataFrame,
    limite_intervalo_s: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Abordagem alternativa: separa por rotulo PRIMEIRO, ordena depois.

    Em `construir_eventos` a ordem e: ordenar o arquivo inteiro por data e depois
    quebrar quando o rotulo muda. Aqui e o contrario: separa as leituras por
    rotulo e so entao ordena cada grupo no tempo.

    A diferenca nao e de estilo. Na primeira, um rotulo que reaparece semanas
    depois vira **dois eventos**, porque no meio houve leituras de outro rotulo.
    Nesta, os dois trechos ficam no **mesmo** evento, ja que dentro do grupo o
    rotulo nunca muda.

    Existe para ser comparada com a outra, nao para substitui-la.
    """
    tempo, rotulo = config.COLUNA_TEMPO, config.COLUNA_ROTULO

    if df.empty:
        vazio = pd.DataFrame(
            columns=["evento", rotulo, "n_leituras", "inicio", "fim",
                     "duracao_s", "duracao_min"]
        )
        return df.assign(evento=pd.Series(dtype="int64")), vazio

    # Ordena por (rotulo, tempo): agrupa primeiro, cronologia depois.
    leituras = df.sort_values([rotulo, tempo], kind="stable").reset_index(drop=True)

    cortes = [segmentos.mudou_valor(leituras[rotulo])]
    if limite_intervalo_s is not None:
        # Dentro de um grupo o tempo e crescente, entao o intervalo faz sentido.
        cortes.append(segmentos.passou_intervalo(leituras[tempo], limite_intervalo_s))

    grupos = segmentos.numerar_grupos(*cortes)
    leituras["evento"] = grupos

    eventos = segmentos.resumir_grupos(
        leituras, grupos, coluna_tempo=tempo, primeiro_de=[rotulo]
    ).rename(columns={"_grupo": "evento"})

    return leituras, eventos


def coesao_eventos(
    leituras: pd.DataFrame, colunas: list[str] | None = None
) -> pd.DataFrame:
    """Mede o quanto as leituras de dentro de cada evento se parecem entre si.

    Um agrupamento so faz sentido se o que ele junta for parecido. Se um evento
    reune leituras muito diferentes, ele juntou coisas que nao deveriam estar
    juntas — e isso da para medir, nao so argumentar.

    Como e feito:

    1. Cada medida de vibracao e padronizada (subtrai a media, divide pelo desvio)
       usando o arquivo INTEIRO como referencia. Sem isso, colunas em escalas
       diferentes — mm/s, graus, Hz — nao poderiam ser somadas.
    2. Para cada evento, calcula-se o ponto medio das suas leituras.
    3. A `dispersao` e a distancia media das leituras ate esse ponto medio.

    Dispersao baixa = leituras parecidas entre si = o evento agrupou bem.
    Dispersao alta  = leituras diferentes no mesmo evento = agrupamento duvidoso.

    A padronizacao usa o arquivo inteiro de proposito: assim o numero de dois
    agrupamentos diferentes pode ser comparado na mesma regua.
    """
    rotulo = config.COLUNA_ROTULO
    colunas = colunas or [c for c in config.COLUNAS_ASSINATURA if c in leituras.columns]
    if not colunas or leituras.empty:
        return pd.DataFrame(columns=["evento", rotulo, "n_leituras", "dispersao"])

    X = leituras[colunas].to_numpy(dtype=float)
    media = np.nanmean(X, axis=0)
    desvio = np.nanstd(X, axis=0)
    desvio[desvio == 0] = 1.0  # coluna constante nao contribui para a distancia
    Z = np.nan_to_num((X - media) / desvio)

    padronizado = pd.DataFrame(Z, columns=colunas)
    padronizado["evento"] = leituras["evento"].to_numpy()

    centro = padronizado.groupby("evento")[colunas].transform("mean").to_numpy()
    distancia = np.sqrt(((Z - centro) ** 2).sum(axis=1))

    saida = pd.DataFrame(
        {
            "evento": leituras["evento"].to_numpy(),
            rotulo: leituras[rotulo].to_numpy(),
            "_d": distancia,
        }
    )
    resultado = (
        saida.groupby("evento")
        .agg(**{rotulo: (rotulo, "first"), "n_leituras": ("_d", "size"),
                "dispersao": ("_d", "mean")})
        .reset_index()
    )
    resultado["dispersao"] = resultado["dispersao"].round(3)
    return resultado


def comparar_abordagens(df: pd.DataFrame) -> dict:
    """Executa as duas abordagens e mede a diferenca.

    Devolve os dois conjuntos de eventos, um resumo lado a lado e a lista dos
    rotulos em que elas discordam — com quantos eventos cada uma produz.
    """
    tempo, rotulo = config.COLUNA_TEMPO, config.COLUNA_ROTULO

    leituras_a, eventos_a = construir_eventos(df)
    leituras_b, eventos_b = construir_eventos_por_rotulo(df)

    buracos_a = segmentos.maior_buraco_interno(leituras_a, leituras_a["evento"], tempo)
    buracos_b = segmentos.maior_buraco_interno(leituras_b, leituras_b["evento"], tempo)

    coesao_a = coesao_eventos(leituras_a)
    coesao_b = coesao_eventos(leituras_b)

    def perfil(eventos, buracos, coesao, nome):
        b = eventos["evento"].map(buracos).fillna(0.0)
        return {
            "abordagem": nome,
            "eventos": int(len(eventos)),
            "duracao_mediana_min": float(eventos["duracao_min"].median()),
            "duracao_maxima_h": float(eventos["duracao_s"].max() / 3600),
            "com_buraco_1h": int((b > 3600).sum()),
            "maior_buraco_h": float(b.max() / 3600),
            "leituras_por_evento": float(eventos["n_leituras"].median()),
            "dispersao_mediana": float(coesao["dispersao"].median()),
            "dispersao_maxima": float(coesao["dispersao"].max()),
        }

    resumo = pd.DataFrame(
        [
            perfil(eventos_a, buracos_a, coesao_a,
                   "A) ordena por data, depois separa por rotulo"),
            perfil(eventos_b, buracos_b, coesao_b,
                   "B) separa por rotulo, depois ordena por data"),
        ]
    )

    por_rotulo = (
        eventos_a.groupby(rotulo, observed=True).size().rename("eventos_A").to_frame()
        .join(eventos_b.groupby(rotulo, observed=True).size().rename("eventos_B"))
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    por_rotulo["diferenca"] = por_rotulo["eventos_A"] - por_rotulo["eventos_B"]
    por_rotulo = por_rotulo.sort_values("diferenca", ascending=False).reset_index(drop=True)

    # As duas concordam? Comparamos QUAIS leituras ficaram juntas, nao quantos
    # eventos sairam: dois agrupamentos podem ter o mesmo total e ainda assim
    # juntar linhas diferentes. Cada evento vira o conjunto dos ids que contem.
    def particao(leituras: pd.DataFrame) -> set:
        por_evento = leituras.groupby("evento")[config.COLUNA_ID].apply(frozenset)
        return set(por_evento)

    mesma_particao = particao(leituras_a) == particao(leituras_b)

    return {
        "eventos_a": eventos_a.merge(
            coesao_a[["evento", "dispersao"]], on="evento", how="left"
        ),
        "eventos_b": eventos_b.merge(
            coesao_b[["evento", "dispersao"]], on="evento", how="left"
        ),
        "leituras_a": leituras_a,
        "leituras_b": leituras_b,
        "coesao_a": coesao_a,
        "coesao_b": coesao_b,
        "resumo": resumo,
        "por_rotulo": por_rotulo,
        "resultado_igual": mesma_particao,
        "rotulos_que_divergem": int((por_rotulo["diferenca"] != 0).sum()),
    }


def diagnostico_ordenacao(df: pd.DataFrame) -> dict:
    """Mede o quanto o arquivo bruto esta fora de ordem.

    `construir_eventos` ordena por data antes de agrupar, e essa etapa nao e
    detalhe: agrupar linhas vizinhas num arquivo desordenado junta leituras que
    nao tem relacao nenhuma entre si.

    Devolve os numeros que justificam a ordenacao, incluindo quantos eventos
    sairiam se ela fosse pulada.
    """
    tempo, rotulo = config.COLUNA_TEMPO, config.COLUNA_ROTULO

    delta = df[tempo].diff().dt.total_seconds()

    # Posicao de cada linha hoje x posicao depois de ordenar.
    posicao_atual = np.arange(len(df))
    posicao_ordenada = df[tempo].rank(method="first").astype("int64").to_numpy() - 1

    eventos_sem_ordenar = (
        int(segmentos.numerar_grupos(segmentos.mudou_valor(df[rotulo]))[-1])
        if len(df)
        else 0
    )

    return {
        "ja_ordenado": bool(df[tempo].is_monotonic_increasing),
        "id_ordenado": bool(df[config.COLUNA_ID].is_monotonic_increasing)
        if config.COLUNA_ID in df.columns
        else None,
        "linhas_que_voltam_no_tempo": int((delta < 0).sum()),
        "maior_salto_para_tras_dias": float(delta.min() / 86400) if len(delta) else 0.0,
        "linhas_fora_do_lugar": int((posicao_atual != posicao_ordenada).sum()),
        "pct_fora_do_lugar": round(
            float((posicao_atual != posicao_ordenada).mean() * 100), 1
        ),
        "eventos_sem_ordenar": eventos_sem_ordenar,
    }


def exemplo_desordem(df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """As duas linhas vizinhas com o maior salto para tras no arquivo bruto.

    Serve de prova concreta: no arquivo como veio, a linha seguinte pode ser de
    um mes antes.
    """
    tempo = config.COLUNA_TEMPO
    delta = df[tempo].diff().dt.total_seconds()
    if delta.isna().all() or delta.min() >= 0:
        return pd.DataFrame()

    i = int(delta.idxmin())
    colunas = [c for c in (config.COLUNA_ID, tempo, config.COLUNA_ROTULO)
               if c in df.columns]
    recorte = df.loc[[i - 1, i], colunas].copy()
    recorte.insert(0, "posicao_no_arquivo", [i - 1, i])
    return recorte


def diagnostico_eventos(
    leituras: pd.DataFrame, eventos: pd.DataFrame, limite_alerta_s: float = 60.0
) -> pd.DataFrame:
    """Aponta eventos que provavelmente deveriam ser mais de um.

    Como a regra atual so quebra na mudanca de rotulo, um evento pode conter uma
    interrupcao longa por dentro: pararam de gravar e retomaram o mesmo ensaio
    dias depois, sem trocar o nome.

    A coluna `maior_buraco_s` mede a maior interrupcao interna. Nada e corrigido
    — a funcao existe para o custo da regra ficar visivel, e nao escondido.
    """
    tempo = config.COLUNA_TEMPO

    buracos = segmentos.maior_buraco_interno(leituras, leituras["evento"], tempo)
    saida = eventos.copy()
    saida["maior_buraco_s"] = saida["evento"].map(buracos).fillna(0.0)
    saida["maior_buraco_h"] = (saida["maior_buraco_s"] / 3600).round(2)
    saida["suspeito"] = saida["maior_buraco_s"] > limite_alerta_s

    return (
        saida[saida["suspeito"]]
        .sort_values("maior_buraco_s", ascending=False)
        .reset_index(drop=True)
    )


def analise_corte_interno(leituras: pd.DataFrame, cortes=None) -> dict:
    """Como um corte por tempo agiria sobre os eventos que ja existem.

    A regra atual so quebra na troca de rotulo, entao alguns eventos carregam
    interrupcoes por dentro. Esta funcao olha **so esses intervalos internos** e
    responde: se passassemos a cortar tambem por tempo, o que mudaria?

    Diferente da analise da tela "Qualidade dos Dados", que examina o arquivo
    inteiro para escolher um limiar. Aqui o recorte ja e o resultado: dos 205
    eventos formados, quais se partiriam e em quantos.

    Devolve:

    - `intervalos`   — todos os intervalos entre leituras dentro de um evento
    - `estatisticas` — minimo, mediana, media e maximo desses intervalos
    - `faixas`       — quantos caem em cada faixa, com marcacao das vazias
    - `vazio`        — a fronteira entre "coleta continua" e "pausa de verdade"
    - `simulacao`    — para cada corte candidato, quantos eventos sairiam
    """
    tempo = config.COLUNA_TEMPO

    delta = leituras[tempo].diff().dt.total_seconds()
    # A primeira linha de cada evento carrega o intervalo em relacao ao evento
    # anterior. Descartamos: so interessa o que acontece por dentro.
    dentro_do_evento = leituras.groupby("evento").cumcount() > 0
    internos = delta.to_numpy()[dentro_do_evento.to_numpy()]
    internos = internos[~pd.isna(internos)]

    if internos.size == 0:
        return {"intervalos": internos, "estatisticas": {}, "faixas": pd.DataFrame(),
                "vazio": {}, "simulacao": pd.DataFrame()}

    estatisticas = {
        "n": int(internos.size),
        "minimo_s": float(internos.min()),
        "mediana_s": float(pd.Series(internos).median()),
        "media_s": float(internos.mean()),
        "maximo_s": float(internos.max()),
    }

    limites = [0, 1, 2.5, 4, 6, 10, 15, 20, 30, 60, 300, 3600, float("inf")]
    faixas = pd.DataFrame(
        {
            "de_s": limites[:-1],
            "ate_s": limites[1:],
            "intervalos": [
                int(((internos >= a) & (internos < b)).sum())
                for a, b in zip(limites[:-1], limites[1:])
            ],
        }
    )
    faixas["vazia"] = faixas["intervalos"] == 0

    LIMITE_CONTINUO = 10.0
    continuos = internos[internos <= LIMITE_CONTINUO]
    pausas = internos[internos > LIMITE_CONTINUO]
    vazio = {
        "maior_continuo_s": float(continuos.max()) if continuos.size else None,
        "menor_pausa_s": float(pausas.min()) if pausas.size else None,
        "n_pausas": int(pausas.size),
    }
    if vazio["maior_continuo_s"] is not None and vazio["menor_pausa_s"] is not None:
        vazio["centro_s"] = (vazio["maior_continuo_s"] + vazio["menor_pausa_s"]) / 2

    # --- simulacao ---------------------------------------------------------
    cortes = list(cortes) if cortes else [2.5, 5, 8, 10, 15, 20, 30, 60, 300, 3600]
    eventos_atuais = int(leituras["evento"].nunique())

    linhas = []
    for c in cortes:
        # Quantos intervalos internos seriam cortados por esse limiar.
        partiria = int((internos > c).sum())
        eventos_partidos = int(
            leituras.assign(_corta=[False] + list(delta.to_numpy()[1:] > c))
            .loc[dentro_do_evento & (delta > c), "evento"]
            .nunique()
        )
        linhas.append(
            {
                "corte_s": c,
                "eventos": eventos_atuais + partiria,
                "eventos_partidos": eventos_partidos,
                "novos_cortes": partiria,
            }
        )

    simulacao = pd.DataFrame(linhas)
    simulacao["pct_eventos_partidos"] = (
        simulacao["eventos_partidos"] / max(eventos_atuais, 1) * 100
    ).round(1)

    return {
        "intervalos": internos,
        "estatisticas": estatisticas,
        "faixas": faixas,
        "vazio": vazio,
        "simulacao": simulacao,
        "eventos_atuais": eventos_atuais,
    }


def criterios_limiar(intervalos: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deriva o limiar de quebra por criterios automaticos, sem escolha visual.

    A defesa de um limiar fica fraca quando ele foi escolhido olhando um grafico.
    Aqui testamos quatro criterios que calculam o numero sozinhos, e reportamos
    tambem os que **nao** funcionam — omitir as falhas seria escolher a dedo o
    criterio que confirma a conclusao.

    Devolve `(criterios, saltos)`:

    - `criterios` — um por linha, com o valor que produz e se e aplicavel
    - `saltos`    — os maiores saltos relativos entre valores consecutivos, que
                    e o criterio que de fato funciona nesta distribuicao
    """
    g = np.asarray(intervalos, dtype=float)
    g = g[~np.isnan(g)]
    if g.size == 0:
        return pd.DataFrame(), pd.DataFrame()

    q1, q3 = np.percentile(g, [25, 75])
    iqr = float(q3 - q1)
    mediana = float(np.median(g))
    mad = float(np.median(np.abs(g - mediana)))

    # Maior salto RELATIVO entre dois valores consecutivos da lista ordenada de
    # valores distintos. Encontra a maior descontinuidade da distribuicao.
    distintos = np.unique(g)
    distintos = distintos[distintos > 0]
    saltos = pd.DataFrame()
    ponto_medio = None
    if distintos.size > 1:
        razao = distintos[1:] / distintos[:-1]
        ordem = np.argsort(razao)[::-1][:5]
        saltos = pd.DataFrame(
            {
                "de_s": distintos[ordem],
                "para_s": distintos[ordem + 1],
                "salto": razao[ordem].round(2),
                "ponto_medio_s": ((distintos[ordem] + distintos[ordem + 1]) / 2).round(3),
            }
        ).reset_index(drop=True)
        maior = int(np.argmax(razao))
        ponto_medio = float((distintos[maior] + distintos[maior + 1]) / 2)

    linhas = [
        {
            "criterio": "Tukey (Q3 + 1,5 x IQR)",
            "valor_s": float(q3 + 1.5 * iqr),
            "observacao": (
                "O criterio de outlier usado no resto do projeto. Depende do IQR, "
                f"que aqui vale {iqr:.5f} s — praticamente zero, porque a esmagadora "
                "maioria das leituras tem exatamente o mesmo intervalo. O limite "
                "desaba em cima da propria cadencia normal."
            ),
        },
        {
            "criterio": "Tukey extremo (Q3 + 3 x IQR)",
            "valor_s": float(q3 + 3.0 * iqr),
            "observacao": "Mesmo problema: tres vezes um IQR quase nulo continua quase nulo.",
        },
        {
            "criterio": "Mediana + 10 x MAD",
            "valor_s": float(mediana + 10 * mad / 0.6745) if mad > 0 else float(mediana),
            "observacao": (
                f"Versao robusta do desvio padrao. Falha igual: o MAD vale "
                f"{mad:.5f} s, pelo mesmo motivo."
            ),
        },
        {
            "criterio": "Percentil 99,8",
            "valor_s": float(np.percentile(g, 99.8)),
            "observacao": (
                "Nao desaba, mas o percentil e escolhido a mao: 99,7 daria 5,5 s e "
                "99,9 daria 55 s. Trocaria um chute por outro."
            ),
        },
        {
            "criterio": "Maior salto relativo",
            "valor_s": ponto_medio if ponto_medio is not None else float("nan"),
            "observacao": (
                "Procura a maior descontinuidade da distribuicao e corta no meio "
                "dela. Nao depende de media, desvio nem percentil escolhido a mao."
            ),
        },
    ]

    criterios = pd.DataFrame(linhas)

    # O veredito nao e um julgamento nosso: e quantos cortes o criterio faria e
    # quantos eventos sairiam. Um criterio que parte a cadencia normal em milhares
    # de pedacos se denuncia sozinho no numero.
    criterios["cortes_que_faria"] = [
        int((g > v).sum()) if np.isfinite(v) else 0 for v in criterios["valor_s"]
    ]

    # Onde exatamente cada limiar cai: qual e o maior intervalo que ele deixa
    # passar e qual e o menor que ele corta. Sao os dois vizinhos do limiar na
    # distribuicao real, e mostram se ele caiu no meio de um grupo ou entre eles.
    maior_que_passa, menor_que_corta = [], []
    for v in criterios["valor_s"]:
        if not np.isfinite(v):
            maior_que_passa.append(np.nan)
            menor_que_corta.append(np.nan)
            continue
        passa = g[g <= v]
        corta = g[g > v]
        maior_que_passa.append(float(passa.max()) if passa.size else np.nan)
        menor_que_corta.append(float(corta.min()) if corta.size else np.nan)

    criterios["maior_que_passa_s"] = np.round(maior_que_passa, 3)
    criterios["menor_que_corta_s"] = np.round(menor_que_corta, 3)
    # Distancia entre esses dois vizinhos. Grande = o limiar caiu num vazio, e
    # move-lo um pouco nao muda nada. Pequena = ele partiu um grupo ao meio.
    criterios["folga_s"] = (
        criterios["menor_que_corta_s"] - criterios["maior_que_passa_s"]
    ).round(3)

    ordem = ["criterio", "valor_s", "maior_que_passa_s", "menor_que_corta_s",
             "folga_s", "cortes_que_faria", "observacao"]
    return criterios[ordem], saltos


def validar_eventos(
    leituras: pd.DataFrame, eventos: pd.DataFrame, original: pd.DataFrame
) -> pd.DataFrame:
    """Checagens que ou passam ou falham, sem espaco para interpretacao.

    Rodam sempre que os eventos sao construidos. Se alguma falhar, o
    agrupamento esta errado e nao adianta olhar o resultado.
    """
    rotulo = config.COLUNA_ROTULO

    rotulos_por_evento = leituras.groupby("evento")[rotulo].nunique()
    soma = int(eventos["n_leituras"].sum())
    ids = leituras["evento"].to_numpy()

    checagens = [
        (
            "Nenhum evento mistura dois rotulos",
            bool((rotulos_por_evento <= 1).all()),
            f"{int((rotulos_por_evento > 1).sum())} evento(s) com mais de um rotulo",
        ),
        (
            "Nenhuma leitura foi perdida ou duplicada",
            soma == len(original),
            f"{soma:,} leituras nos eventos vs {len(original):,} no arquivo".replace(",", "."),
        ),
        (
            "Toda leitura pertence a um evento",
            bool(leituras["evento"].notna().all()),
            f"{int(leituras['evento'].isna().sum())} leitura(s) sem evento",
        ),
        (
            "As leituras estao em ordem de data",
            bool(leituras[config.COLUNA_TEMPO].is_monotonic_increasing),
            "ordenacao crescente por created_at",
        ),
        (
            "Os numeros de evento sao consecutivos",
            bool(np.array_equal(np.unique(ids), np.arange(1, len(eventos) + 1))),
            f"{len(eventos)} eventos, de {ids.min()} a {ids.max()}",
        ),
    ]

    return pd.DataFrame(checagens, columns=["checagem", "passou", "detalhe"])


def resumo_por_rotulo(eventos: pd.DataFrame) -> pd.DataFrame:
    """Quantos eventos e quanto tempo cada rotulo acumula.

    E a tabela que responde a pergunta do operador: nao "quantas linhas", mas
    "quantas vezes".
    """
    rotulo = config.COLUNA_ROTULO

    resumo = (
        eventos.groupby(rotulo, observed=True)
        .agg(
            eventos=("evento", "size"),
            leituras=("n_leituras", "sum"),
            duracao_mediana_min=("duracao_min", "median"),
            duracao_total_h=("duracao_s", lambda s: s.sum() / 3600),
            primeira=("inicio", "min"),
            ultima=("fim", "max"),
        )
        .reset_index()
    )
    resumo["leituras_por_evento"] = (resumo["leituras"] / resumo["eventos"]).round(0)
    return resumo.sort_values("leituras", ascending=False).reset_index(drop=True)
