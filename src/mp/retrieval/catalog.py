"""Catalogo de falhas — leitura do `data/fault_map.yaml`.

Este modulo e a implementacao do principio 1 do projeto: numero nunca e
comparado com texto. O caminho e sempre

    rotulo cru  ->  familia  ->  documento

e cada seta e um lookup exato, nunca uma similaridade. E aqui que os guardrails
**G2** (rotulo e estado, nao problema) e **G3** (familia tem documento?) buscam
a resposta.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from mp import config

# --------------------------------------------------------------------------
# As quatro situacoes possiveis
# --------------------------------------------------------------------------
#
# Do ponto de vista do fluxo, as tres ultimas terminam igual: nao ha prescricao
# e o modelo nao e chamado. Sao nomeadas separadamente porque sao **noticias
# diferentes** para quem le: dizer "sem documentacao" quando a maquina esta
# apenas normal seria mentir, e nao ha nada a registrar.
SITUACAO_OK = "ok"                        # defeito com manual — prescreve
SITUACAO_ESTADO = "estado"                # normal, teste... nao ha o que corrigir
SITUACAO_SEM_DOCUMENTO = "sem_documento"  # e defeito, mas falta o manual
SITUACAO_DESCONHECIDO = "desconhecido"    # rotulo fora do catalogo


@functools.lru_cache(maxsize=4)
def carregar_fault_map(caminho: str | Path | None = None) -> dict:
    """Le o YAML. Cacheado — o arquivo e pequeno e nao muda em runtime."""
    caminho = Path(caminho) if caminho else config.FAULT_MAP_PATH
    if not caminho.exists():
        raise FileNotFoundError(
            f"{caminho} nao existe. Ele e versionado — se sumiu, restaure do git."
        )
    with open(caminho, encoding="utf-8") as fh:
        mapa = yaml.safe_load(fh)

    if "familias" not in mapa:
        raise ValueError(f"{caminho} nao tem a chave 'familias'.")
    return mapa


@functools.lru_cache(maxsize=4)
def _indice_aliases(caminho: str | None = None) -> dict[str, str]:
    """Indice reverso alias -> familia, construido uma vez.

    Levanta erro se o mesmo alias aparecer em duas familias: seria ambiguidade
    silenciosa no ponto mais critico do sistema.
    """
    mapa = carregar_fault_map(caminho)
    indice: dict[str, str] = {}
    for familia, dados in mapa["familias"].items():
        for alias in dados.get("aliases", []):
            if alias in indice and indice[alias] != familia:
                raise ValueError(
                    f"Alias '{alias}' aparece em '{indice[alias]}' e em '{familia}'. "
                    "Um rotulo so pode pertencer a uma familia."
                )
            indice[alias] = familia
    return indice


def familia_de(rotulo: str, caminho: str | None = None) -> str | None:
    """Familia de um rotulo cru. `None` se o rotulo nao esta no catalogo."""
    return _indice_aliases(caminho).get(str(rotulo).strip())


def is_problem(familia: str, caminho: str | None = None) -> bool | None:
    """Se a familia e defeito (True) ou estado (False). Base do **G2**."""
    dados = carregar_fault_map(caminho)["familias"].get(familia)
    return None if dados is None else bool(dados.get("is_problem", False))


def documentos_de(familia: str, caminho: str | None = None) -> list[dict]:
    """Documentos que cobrem a familia. Base do **G3**.

    Lista vazia significa recusa — inclusive quando a cobertura e `parcial`, que
    e o caso do `eccentric_rotor`. Cobertura parcial nao autoriza prescricao.
    """
    dados = carregar_fault_map(caminho)["familias"].get(familia)
    return list(dados.get("documentos") or []) if dados else []


def familias_do_documento(documento_id: str, caminho: str | None = None) -> list[str]:
    """Quais familias um documento cobre. E o caminho inverso do G3.

    Existe para o caso em que o tecnico chega descrevendo o problema por escrito:
    ali quem aparece primeiro e o **documento**, e a familia e deduzida dele.
    A ligacao e a mesma do `fault_map.yaml`, so lida ao contrario — nao ha
    segunda fonte de verdade.
    """
    mapa = carregar_fault_map(caminho)["familias"]
    return [
        familia
        for familia, dados in mapa.items()
        if any(d.get("id") == documento_id for d in (dados.get("documentos") or []))
    ]


@dataclass(frozen=True)
class Catalogo:
    """O percurso `rotulo -> familia -> documento`, ja com o veredito.

    Substitui o dicionario de nove chaves que a versao anterior devolvia. A
    decisao esta em `situacao`; `e_defeito` e `prescrever` sao leituras dela,
    nao campos independentes que poderiam divergir.
    """

    rotulo: str
    familia: str | None
    documentos: list[dict]
    situacao: str
    mensagem: str

    @property
    def e_defeito(self) -> bool:
        """Veredito do **G2**. Defeito passa, com manual ou sem."""
        return self.situacao in (SITUACAO_OK, SITUACAO_SEM_DOCUMENTO)

    @property
    def prescrever(self) -> bool:
        """Veredito do **G3**. So o defeito com manual autoriza prescricao."""
        return self.situacao == SITUACAO_OK

    @property
    def documento_ids(self) -> list[str]:
        return [d["id"] for d in self.documentos]


def verificar_existencia_conserto(
    rotulo_ou_familia: str | None, caminho: str | None = None
) -> Catalogo:
    """Responde uma pergunta so: **chegou este rotulo — da para prescrever conserto?**

    Recebe um rotulo cru (`"cocked_rotor_2"`) ou o nome da familia, e devolve um
    `Catalogo`:

        c = verificar_existencia_conserto("cocked_rotor_2")

        c.familia        # "cocked_rotor"
        c.situacao       # "ok"
        c.documentos     # [{"id": "Doc6", ...}]
        c.mensagem       # "1 documento(s): Doc6."
        c.prescrever     # True

    Sao tres perguntas em sequencia, todas consulta exata ao `fault_map.yaml`:

    1. Que familia e essa?  Nao achou      -> `desconhecido`
    2. E defeito?           normal, teste  -> `estado`
    3. Tem manual?          lista vazia    -> `sem_documento`

    Chegou ao fim das tres -> `ok`.

    Ponto de entrada unico do catalogo: e daqui que o G2 e o G3 tiram a resposta,
    em vez de reabrir o YAML cada um por sua conta. O nome diz o que a funcao
    decide — se existe conserto documentado —, nao como ela chega la.
    """
    if not rotulo_ou_familia:
        return Catalogo("", None, [], SITUACAO_DESCONHECIDO,
                        "Sem rotulo para consultar o catalogo.")

    rotulo = str(rotulo_ou_familia).strip()
    familias = carregar_fault_map(caminho)["familias"]
    familia = rotulo if rotulo in familias else familia_de(rotulo, caminho)

    if familia is None:
        return Catalogo(
            rotulo, None, [], SITUACAO_DESCONHECIDO,
            f"'{rotulo}' nao esta no catalogo — registre-o no fault_map.yaml.",
        )

    dados = familias[familia]
    docs = list(dados.get("documentos") or [])

    if not dados.get("is_problem", False):
        return Catalogo(
            rotulo, familia, docs, SITUACAO_ESTADO,
            f"'{familia}' e um estado da maquina, nao um defeito. "
            "Nao ha acao corretiva a prescrever.",
        )

    if not docs:
        # Cobertura parcial conta como ausencia — `eccentric_rotor` aparece no
        # manual de polias, mas e excentricidade de polia, nao de rotor.
        parcial = dados.get("cobertura") == "parcial"
        return Catalogo(
            rotulo, familia, [], SITUACAO_SEM_DOCUMENTO,
            f"Sem documentacao para '{familia}' — registre um documento."
            + (f" (cobertura parcial em {dados.get('cobertura_parcial_em')}, "
               "insuficiente para prescrever)" if parcial else ""),
        )

    ids = ", ".join(d["id"] for d in docs)
    return Catalogo(rotulo, familia, docs, SITUACAO_OK,
                    f"{len(docs)} documento(s): {ids}.")


def tabela_familias(caminho: str | None = None) -> pd.DataFrame:
    """O catalogo inteiro como DataFrame — para a UI e para os notebooks."""
    mapa = carregar_fault_map(caminho)
    linhas = []
    for familia, dados in mapa["familias"].items():
        docs = dados.get("documentos") or []
        linhas.append(
            {
                "familia": familia,
                "descricao": dados.get("descricao", ""),
                "is_problem": bool(dados.get("is_problem", False)),
                "cobertura": dados.get("cobertura", ""),
                "documentos": ", ".join(d["id"] for d in docs),
                "n_rotulos": int(dados.get("n_rotulos", len(dados.get("aliases", [])))),
                "n_leituras": int(dados.get("n_leituras", 0)),
                "g3_libera": bool(dados.get("is_problem", False)) and bool(docs),
            }
        )
    return (
        pd.DataFrame(linhas)
        .sort_values("n_leituras", ascending=False)
        .reset_index(drop=True)
    )


def validar_cobertura(df: pd.DataFrame, caminho: str | None = None) -> dict:
    """Confere que o catalogo cobre todo rotulo presente no DataFrame.

    Um rotulo fora do catalogo nao quebra o pipeline — a verificacao devolve
    recusa —, mas e sempre um erro de curadoria: alguem coletou uma condicao
    nova e ninguem registrou. Por isso a checagem e explicita.
    """
    presentes = set(df[config.COLUNA_ROTULO].dropna().astype(str).unique())
    indice = _indice_aliases(caminho)
    catalogados = set(indice)

    faltando = sorted(presentes - catalogados)
    sobrando = sorted(catalogados - presentes)

    return {
        "rotulos_no_dado": len(presentes),
        "rotulos_no_catalogo": len(catalogados),
        "sem_familia": faltando,
        "no_catalogo_sem_dado": sobrando,
        "ok": not faltando,
    }
