"""O pipeline inteiro executado passo a passo, cronometrado e observavel.

Os outros modulos deste pacote sao as pecas; este as roda em ordem e conta o
que aconteceu em cada uma. Serve ao painel de execucao da interface, e serve
igualmente no terminal — `python -m mp.classificacao.execucao` imprime o mesmo
relatorio, sem Streamlit nenhum.

**Por que um modulo, e nao um laco dentro da tela.** A sequencia das etapas e o
que cada uma produz sao conhecimento do dominio, nao de desenho: quem le o
codigo deve conseguir saber o que o sistema faz sem abrir um arquivo de
interface. A tela recebe `Etapa` e desenha; nao decide o que vem depois de que.

**A cronometragem e honesta, e isso custa uma decisao.** As funcoes rodam aqui
sem passar pelo cache da interface — senao a segunda execucao mostraria tempos
proximos de zero e o painel viraria enfeite. Executar de novo executa de novo.

Cada etapa entrega tambem uma amostra pequena do que produziu. E o que
transforma o painel de barra de progresso em prova: da para ver o dado mudando
de forma entre uma etapa e a seguinte, em vez de acreditar que mudou.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mp import config
from mp.analysis import carregar
from mp.classificacao import amostras as A
from mp.classificacao.modelo import Classificador
from mp.classificacao.validacao import validar

__all__ = ["Etapa", "executar_pipeline", "relatorio", "relatorio_texto"]

LINHAS_DE_AMOSTRA = 6


@dataclass
class Etapa:
    """O registro de uma etapa que ja terminou.

    `resultado` so vem preenchido na etapa de validacao — e o dicionario que
    `validacao.validar` devolve, com folds, matriz e metricas. As demais etapas
    o deixam vazio: nem toda etapa produz um veredito.
    """

    numero: int
    total: int
    nome: str
    o_que_faz: str
    onde: str
    segundos: float
    resumo: dict[str, str] = field(default_factory=dict)
    amostra: pd.DataFrame | None = None
    resultado: dict | None = None

    @property
    def rotulo(self) -> str:
        return f"{self.numero}/{self.total} · {self.nome}"


def _cronometrar(funcao, *args, **kwargs):
    """Roda e devolve `(valor, segundos)`. `perf_counter` por ser monotonico."""
    inicio = time.perf_counter()
    valor = funcao(*args, **kwargs)
    return valor, time.perf_counter() - inicio


def _milhar(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def executar_pipeline(
    tamanho: int | None = None,
    incluir_regime: bool = False,
    so_defeitos: bool = False,
    n_arvores: int | None = None,
    n_folds: int | None = None,
) -> Iterator[Etapa]:
    """Roda as cinco etapas, entregando cada uma assim que ela termina.

    Gerador de proposito: quem consome desenha a etapa concluida enquanto a
    seguinte ainda roda. Devolver a lista pronta no fim daria a mesma
    informacao 30 segundos depois, com a tela parada no meio.

    As cinco etapas sao as mesmas funcoes que qualquer outro consumidor do
    pacote chama — nada aqui e um caminho paralelo.
    """
    tamanho = tamanho or config.CLF_JANELA_TAMANHO
    n_arvores = n_arvores or config.CLF_N_ARVORES
    n_folds = n_folds or config.CLF_N_FOLDS
    total = 5

    # -- 1. o arquivo ------------------------------------------------------
    bruto, seg = _cronometrar(carregar)
    yield Etapa(
        numero=1,
        total=total,
        nome="Ler o arquivo bruto",
        o_que_faz="Le o `banner.csv`, tipa `created_at` e apara `fault`. Nao "
                  "ordena, nao deduplica, nao descarta coluna.",
        onde="mp.analysis.loader.carregar",
        segundos=seg,
        resumo={
            "linhas": _milhar(len(bruto)),
            "colunas": str(bruto.shape[1]),
            "rotulos crus distintos": str(bruto[config.COLUNA_ROTULO].nunique()),
            "celulas vazias": _milhar(int(bruto.isna().sum().sum())),
        },
        amostra=_colunas_se_houver(
            bruto,
            [config.COLUNA_ID, config.COLUNA_TEMPO, config.COLUNA_ROTULO,
             "rpm", "z_rms_velocity_mm_s", "z_kurtosis"],
        ),
    )

    # -- 2. familia e evento ----------------------------------------------
    leituras, seg = _cronometrar(A.preparar, bruto, so_defeitos=so_defeitos)
    yield Etapa(
        numero=2,
        total=total,
        nome="Resolver a familia e agrupar em eventos",
        o_que_faz="Ordena no tempo, quebra em eventos por `fault` + `rpm`, e "
                  "resolve cada rotulo cru para a familia do `fault_map.yaml`. "
                  "Duas colunas novas: `evento` e `familia`.",
        onde="mp.classificacao.amostras.preparar",
        segundos=seg,
        resumo={
            "leituras mantidas": _milhar(len(leituras)),
            "rotulos crus -> familias":
                f"{bruto[config.COLUNA_ROTULO].nunique()} -> "
                f"{leituras['familia'].nunique()}",
            "eventos formados": str(leituras["evento"].nunique()),
            "leituras sem familia no catalogo": _milhar(len(bruto) - len(leituras)),
        },
        amostra=_colunas_se_houver(
            leituras,
            [config.COLUNA_TEMPO, config.COLUNA_ROTULO, "familia", "evento", "rpm"],
        ),
    )

    # -- 3. as amostras ----------------------------------------------------
    tabela, seg = _cronometrar(
        A.criar_amostras, leituras, modo="janela", tamanho=tamanho,
        incluir_regime=incluir_regime,
    )
    colunas = A.colunas_de_entrada(leituras, incluir_regime=incluir_regime)
    nomes = A.nomes_das_features(colunas)
    X, y, grupos = A.matriz(tabela)

    eventos_totais = leituras["evento"].nunique()
    eventos_usados = int(tabela["evento"].nunique())
    yield Etapa(
        numero=3,
        total=total,
        nome="Recortar em janelas e resumir",
        o_que_faz=f"Blocos de {tamanho} leituras consecutivas dentro de cada "
                  f"evento, com {tamanho // 2} de passo, e cada bloco resumido "
                  f"em {len(A.ESTATISTICAS)} numeros por coluna. Evento mais "
                  "curto que a janela e descartado inteiro.",
        onde="mp.classificacao.amostras.criar_amostras",
        segundos=seg,
        resumo={
            "matriz": f"{_milhar(X.shape[0])} x {X.shape[1]}",
            "colunas de medida usadas": str(len(colunas)),
            "eventos aproveitados": f"{eventos_usados} de {eventos_totais}",
            "familias no conjunto": str(int(tabela["familia"].nunique())),
        },
        amostra=_previa_da_matriz(X, y, grupos, nomes),
    )

    # -- 4. o treino -------------------------------------------------------
    modelo = Classificador(
        tamanho=tamanho, incluir_regime=incluir_regime,
        so_defeitos=so_defeitos, n_arvores=n_arvores,
    )
    _, seg = _cronometrar(modelo.ajustar, tabela)
    yield Etapa(
        numero=4,
        total=total,
        nome="Treinar a floresta",
        o_que_faz=f"Ajusta {n_arvores} arvores de decisao sobre a matriz "
                  "inteira. Este modelo serve para inspecionar em que o "
                  "aprendizado se apoia — **nao** para medir acuracia, porque "
                  "ele viu tudo.",
        onde="mp.classificacao.modelo.Classificador.ajustar",
        segundos=seg,
        resumo={
            "arvores": str(n_arvores),
            "amostras de treino": _milhar(modelo.n_amostras),
            "familias que ele sabe nomear": str(len(modelo.classes_)),
            "features": str(len(modelo.nomes_features)),
        },
        amostra=modelo.importancia_por_coluna().head(LINHAS_DE_AMOSTRA),
    )

    # -- 5. a validacao ----------------------------------------------------
    resultado, seg = _cronometrar(
        validar, tabela, n_arvores=n_arvores, n_folds=n_folds
    )
    # `validar` recebe amostras prontas e nao sabe de que janela elas sairam.
    # Quem sabe e esta funcao, e o relatorio precisa do numero para o resultado
    # ser reproduzivel a partir do que esta escrito nele.
    resultado["janela"] = tamanho
    resultado["incluir_regime"] = incluir_regime
    aleatoria = resultado["estrategias"]["aleatoria"]
    por_evento = resultado["estrategias"]["por_evento"]
    yield Etapa(
        numero=5,
        total=total,
        nome="Validar com as duas estrategias",
        o_que_faz=f"Treina {2 * n_folds} florestas novas: {n_folds} folds "
                  f"sorteando amostras soltas e {n_folds} sorteando eventos "
                  "inteiros. So a segunda responde 'funciona numa maquina nova?'.",
        onde="mp.classificacao.validacao.validar",
        segundos=seg,
        resumo={
            "acuracia honesta (por evento)": f"{por_evento['acuracia']:.1%}",
            "acuracia inflada (aleatoria)": f"{aleatoria['acuracia']:.1%}",
            "inflacao": f"{resultado['inflacao']:.1%}",
            "eventos vazados (aleatoria)": str(aleatoria["eventos_vazados"]),
        },
        amostra=_previa_dos_folds(resultado),
        resultado=resultado,
    )


def _colunas_se_houver(df: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    """Preview com as colunas pedidas que existirem.

    A previa e ilustracao, nao contrato: se um dia uma dessas colunas mudar de
    nome, o painel deve mostrar as outras e seguir, e nao derrubar a execucao
    inteira por causa de uma vitrine.
    """
    presentes = [c for c in colunas if c in df.columns]
    return df.head(LINHAS_DE_AMOSTRA)[presentes]


def _previa_da_matriz(
    X: np.ndarray, y: np.ndarray, grupos: np.ndarray, nomes: list[str]
) -> pd.DataFrame:
    """As primeiras linhas e colunas da matriz, com a classe e o grupo a vista."""
    n = min(LINHAS_DE_AMOSTRA, len(X))
    previa = pd.DataFrame(X[:n, :5], columns=nomes[:5]).round(3)
    previa.insert(0, "familia", y[:n])
    previa.insert(1, "evento", grupos[:n])
    return previa


def _previa_dos_folds(resultado: dict) -> pd.DataFrame:
    """Uma linha por fold das duas estrategias, com a prova do vazamento."""
    linhas = []
    for nome, dados in resultado["estrategias"].items():
        for f in dados["folds"]:
            linhas.append(
                {
                    "estrategia": nome,
                    "fold": f["fold"],
                    "treino": f["n_treino"],
                    "teste": f["n_teste"],
                    "eventos nos dois lados": f["eventos_vazados"],
                    "acuracia": round(f["acuracia"], 4),
                }
            )
    return pd.DataFrame(linhas)


def relatorio(**kwargs) -> dict:
    """Roda o pipeline inteiro e devolve `(etapas, resultado, segundos)`.

    Atalho para quem nao quer consumir o gerador. A interface usa o gerador,
    porque ela desenha enquanto roda.
    """
    etapas = list(executar_pipeline(**kwargs))
    return {
        "etapas": etapas,
        "resultado": etapas[-1].resultado,
        "segundos": sum(e.segundos for e in etapas),
    }


def relatorio_texto(etapas: list[Etapa]) -> str:
    """O relatorio em texto puro, para o terminal e para o botao de baixar.

    Uma formatacao so para os dois: o arquivo que a tela oferece e literalmente
    o que o terminal imprime, e nao uma segunda versao que poderia divergir.
    """
    largura = 62
    linhas = [
        "RELATORIO DE EXECUCAO — classificacao de familia por vibracao",
        "=" * largura,
        "",
        f"{'etapa':<46}{'tempo':>16}",
        "-" * largura,
    ]

    for etapa in etapas:
        linhas.append(f"{etapa.rotulo:<46}{etapa.segundos:>15.1f}s")
        for chave, valor in etapa.resumo.items():
            linhas.append(f"    {chave:.<42} {valor}")

    linhas += [
        "-" * largura,
        f"{'TOTAL':<46}{sum(e.segundos for e in etapas):>15.1f}s",
    ]

    r = next((e.resultado for e in reversed(etapas) if e.resultado), None)
    if r is None:
        return "\n".join(linhas)

    linhas += [
        "",
        "RESULTADO DOS TESTES",
        "=" * largura,
        f"  janela {r.get('janela', config.CLF_JANELA_TAMANHO)} leituras | "
        f"{r['n_arvores']} arvores | {r['n_folds']} folds",
        f"  amostras {_milhar(r['n_amostras'])} | eventos {r['n_eventos']} | "
        f"familias {r['n_familias']}",
        "",
    ]
    for nome, dados in r["estrategias"].items():
        veredito = "HONESTO" if nome == "por_evento" else "INFLADO "
        linhas.append(
            f"  {nome:<12} {dados['acuracia']:>6.1%}  (+-{dados['desvio']:.1%})"
            f"   F1 {dados['f1_macro']:>5.1%}"
            f"   vazados: {dados['eventos_vazados']:<4} {veredito}"
        )

    base = r["linha_de_base"]
    linhas += [
        "",
        f"  inflacao ................ {r['inflacao']:.1%}",
        f"  chutar no acaso ......... {base['aleatorio']:.1%}",
        f"  responder sempre '{base['familia_mais_comum']}' ... {base['maioria']:.1%}",
    ]
    if r["classes_raras"]:
        linhas.append(
            f"  fora da validacao ....... {', '.join(r['classes_raras'])} "
            f"(menos de {r['n_folds']} eventos)"
        )

    linhas += ["", "DETALHE DOS FOLDS", "-" * largura]
    for nome, dados in r["estrategias"].items():
        linhas.append(f"  {nome}")
        for f in dados["folds"]:
            linhas.append(
                f"    fold {f['fold']}  treino {f['n_treino']:>5}  "
                f"teste {f['n_teste']:>5}  "
                f"vazados {f['eventos_vazados']:>4}  "
                f"acuracia {f['acuracia']:.1%}"
            )

    return "\n".join(linhas)


if __name__ == "__main__":  # pragma: no cover
    print(relatorio_texto(list(executar_pipeline())))
