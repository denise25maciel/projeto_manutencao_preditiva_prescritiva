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
from pathlib import Path

import pandas as pd
import yaml

from mp import config


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


def resolver(rotulo: str, caminho: str | None = None) -> dict:
    """Percurso completo rotulo -> familia -> documento, com o veredito dos guardrails.

    Ponto de entrada unico para o pipeline. Devolve tudo que os guardrails
    precisam, sem que eles tenham de reabrir o YAML.
    """
    familia = familia_de(rotulo, caminho)

    if familia is None:
        return {
            "rotulo": rotulo, "familia": None, "conhecido": False,
            "is_problem": None, "documentos": [], "cobertura": None,
            "g2_prossegue": False, "g3_prossegue": False,
            "motivo": "Rotulo fora do catalogo — registre-o no fault_map.yaml.",
        }

    dados = carregar_fault_map(caminho)["familias"][familia]
    problema = bool(dados.get("is_problem", False))
    docs = list(dados.get("documentos") or [])
    cobertura = dados.get("cobertura")

    if not problema:
        motivo = (
            f"'{familia}' e um estado da maquina, nao um defeito. "
            "Nao ha acao corretiva a prescrever."
        )
    elif not docs:
        motivo = (
            f"Sem documentacao para '{familia}' — registre um documento."
            + (f" (cobertura {cobertura} em {dados.get('cobertura_parcial_em')}, "
               "insuficiente para prescrever)" if cobertura == "parcial" else "")
        )
    else:
        motivo = f"'{familia}' tem {len(docs)} documento(s) no catalogo."

    return {
        "rotulo": rotulo, "familia": familia, "conhecido": True,
        "is_problem": problema, "documentos": docs, "cobertura": cobertura,
        "g2_prossegue": problema,
        "g3_prossegue": problema and bool(docs),
        "motivo": motivo,
    }


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

    Um rotulo fora do catalogo nao quebra o pipeline — `resolver` devolve
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
