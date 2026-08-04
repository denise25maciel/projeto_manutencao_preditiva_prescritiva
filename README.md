# Manutenção Preditiva e Prescritiva

Sistema de manutenção prescritiva para máquinas rotativas de chão de fábrica.

Recebe um evento JSON de sensor de vibração, encontra ocorrências similares no histórico
e, se houver documentação para o defeito identificado, sugere a ação corretiva **citando
a fonte**.

A solução não depende de classificação prévia de falhas conhecidas: parte da
identificação de padrões similares dentro do histórico operacional, combinando análise
de dados, busca por similaridade e recuperação de conhecimento (RAG) para apoiar a
decisão das equipes técnicas.

## Estado atual

| Parte | Escopo | Status |
|---|---|---|
| 0 | Módulo de análise exploratória | ✅ concluída |
| 1 | Tratamento, episódios e `fault_map.yaml` | ⬜ |
| 2 | Banco SQLite via SQLAlchemy | ⬜ |
| 3 | Motor de similaridade (kNN) | ⬜ |
| 4 | RAG e guardrails G0–G5 | ⬜ |
| 5 | LLM local (Ollama) | ⬜ |
| 6 | API FastAPI + deploy | ⬜ |

## Configuração do ambiente

Requer **Python 3.11+**. Ambiente validado: Python 3.14.4 no Windows 11.

### 1. Clonar e entrar no diretório

```bash
git clone <url-do-repositorio>
cd projeto_manutencao_preditiva_prescritiva
```

### 2. Criar e ativar o ambiente virtual

O venv isola as dependências do projeto do Python do sistema. `.venv/` está no
`.gitignore` — cada máquina cria o seu.

```powershell
# Windows — PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# Windows — Git Bash
python -m venv .venv
source .venv/Scripts/activate
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

> Se o PowerShell recusar a ativação com `execution of scripts is disabled`, rode
> uma vez: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

Com o venv ativo o prompt fica prefixado com `(.venv)`. Confira com:

```bash
python -c "import sys; print(sys.prefix)"   # deve apontar para .../.venv
```

### 3. Instalar as dependências

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

O `requirements.txt` cobre o que a Parte 0 precisa: pandas, numpy, pyarrow,
streamlit, altair, pyyaml — mais jupyterlab e matplotlib para os notebooks.
As bibliotecas das partes seguintes (SQLAlchemy, scikit-learn, FastAPI, Ollama)
estão listadas em comentário no fim do arquivo e mapeadas como extras no
`pyproject.toml`; entram quando cada parte for implementada.

> O import de `src/mp/` funciona sem instalar o pacote — `ui/_dados.py` e o
> notebook resolvem o caminho sozinhos. Se preferir o pacote instalado, use
> `pip install -e .` (opcional).

### 4. Disponibilizar o dataset

O dataset bruto **não está no repositório** — é dado da empresa, bloqueado pelo
`.gitignore`. Coloque o `banner.csv` em `data/raw/`:

```bash
mkdir -p data/raw
cp /caminho/para/banner.csv data/raw/
```

Ou aponte a variável de ambiente, sem mover o arquivo:

```bash
export MP_CSV=/caminho/para/banner.csv       # Linux/macOS/Git Bash
```
```powershell
$env:MP_CSV = "C:\caminho\para\banner.csv"   # PowerShell
```

O [`src/mp/config.py`](src/mp/config.py) procura, nesta ordem: `MP_CSV`,
`data/raw/`, `docs/`, `descricao_desafio/`, `teste_industria/`. Se não achar,
a UI mostra a mensagem explicando onde colocar em vez de quebrar.

## Como executar

Com o venv ativo, a partir da **raiz do projeto**:

### Interface Streamlit

```bash
streamlit run ui/app.py
```

Abre em `http://localhost:8501`. Três telas no menu lateral:

| Tela | O que mostra |
|---|---|
| **Visão geral** | Números de cabeçalho e os três achados que contrariam a suposição inicial |
| **Análise de Falhas** | Valores únicos de `fault` com busca e filtro. Ao selecionar um rótulo: assinatura de vibração com quartis e CV, o que o distingue do resto do dataset, distribuição de cada feature e outliers dentro da classe |
| **Qualidade dos Dados** | Como o dado chegou: nulos por coluna, cadência de coleta, colunas constantes e redundantes, duplicatas e outliers |

O primeiro carregamento lê o CSV (~0,4 s para 166 mil linhas) e guarda em
`@st.cache_data`; os cliques seguintes não releem o arquivo.

### Notebook

```bash
jupyter lab notebooks/01_eda.ipynb
```

Ou execute sem abrir a interface:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --stdout > /dev/null
```

### Usar o módulo direto

```python
import sys; sys.path.insert(0, "src")

from mp.analysis import carregar, perfil_rotulos, assinaturas_por_rotulo

df = carregar()
perfil_rotulos(df)                          # rótulos, contagem, janela temporal
assinaturas_por_rotulo(df, min_leituras=100)  # tabela de assinaturas
```

## Arquitetura

### Princípios

1. **A coluna `fault` é a chave de junção.** Números nunca são comparados com texto. O
   evento resolve para um rótulo; o rótulo resolve para um documento via
   `data/fault_map.yaml`, curado à mão e versionado.
2. **Fronteira determinística/generativa.** Toda pergunta respondível por consulta ao
   banco não passa pelo LLM. O modelo recebe números prontos; nunca os produz.
3. **Guardrails são código, não prompt.** Verificações determinísticas G0–G5, na ordem.
4. **Busca vetorial nunca retorna vazio** — por isso G3 é um `SELECT` no catálogo, nunca
   uma similaridade semântica.
5. **Sem duplicação de lógica.** Notebooks e UI importam `src/mp/`. Nunca reimplementam.

### Notebooks não fazem parte do runtime

Os notebooks em `notebooks/` **documentam decisões** — o porquê de cada limiar, cada
descarte de coluna, cada escolha de agregação. Eles importam `src/mp/` e não
reimplementam nada. Nada em produção depende deles.

### Estrutura

```
├── data/
│   ├── raw/                  # fora do git — dado da empresa
│   ├── mp.db                 # fora do git
│   └── fault_map.yaml        # VERSIONADO — é decisão, não dado
├── notebooks/01_eda.ipynb
├── src/mp/
│   ├── config.py             # todo limiar e caminho
│   └── analysis/             # loader, profiling, quality, signatures
├── ui/
│   ├── app.py                # streamlit run ui/app.py
│   ├── _dados.py             # ponte cacheada UI -> mp.analysis
│   └── pages/
├── requirements.txt          # instalação do ambiente
└── pyproject.toml            # metadados + extras das partes seguintes
```

> **Nota de arquitetura.** A partir da Parte 5 a UI passa a falar com a API por HTTP,
> para o rerun do Streamlit não recarregar o LLM a cada clique. Na Parte 0 não há modelo
> nenhum — é tudo pandas, cacheável com `@st.cache_data` — então o import direto de
> `src/mp/` é o caminho simples. `ui/_dados.py` isola essa dependência num arquivo só,
> que é o que vai mudar depois.

## Parte 0 — resultados

166.796 leituras × 26 colunas, coletadas entre 30/04 e 16/06/2026.

### Três achados que contrariam a suposição inicial

**1. `created_at` não está em ordem cronológica.** Há saltos negativos de dezenas de
dias entre linhas vizinhas: são ~331 sessões gravadas em épocas diferentes e
concatenadas fora de ordem.

> **Consequência:** toda operação que depende de vizinhança temporal — a mediana móvel
> da Parte 3, a formação de episódios da Parte 1 — precisa ordenar por `created_at`
> antes, e nunca atravessar a fronteira entre sessões.

**2. `z_peak_vel_comp_freq_hz` e `x_peak_vel_comp_freq_hz` não são constantes em 61 Hz.**
Têm 79 e 50 valores distintos. 61 Hz é a moda (60% e 49% das linhas), não o valor único.
As colunas carregam informação e **não devem ser descartadas** — a frequência do pico se
desloca justamente em alguns defeitos.

**3. São 151 rótulos distintos, não ~10.** A inflação vem de erros de digitação
(`mortor_desligado_novo`, `normla_carga_3_3`, `cockecocked_adxl_0`), sufixos de
sessão (`_2`, `_pos_2`, `_carga`, `_adxl_0`) e do prefixo `new_`. O heurístico de
`sugerir_familias` consolida os 151 em **16 famílias**, sem sobra — mas é sugestão: a
decisão vira o `fault_map.yaml` curado à mão na Parte 1.

### Confirmado

- **Zero nulos declarados** nas 26 colunas
- **Cadência de ~2 s** — intervalo mediano 2,0 s, 92% das leituras dentro de ±0,25 s
- **Unidades duplicadas** — `mm/s = in/s × 25,4` e `°F = °C × 9/5 + 32`, com erro máximo
  na casa do arredondamento do arquivo. Confirmado por identidade numérica, não por
  correlação
- **5,8% de duplicatas consecutivas** — 9.736 linhas idênticas à anterior em todas as
  colunas de medida

### Colunas a descartar

| Coluna | Motivo |
|---|---|
| `z_rms_velocity_in_s`, `x_rms_velocity_in_s`, `z_peak_velocity_in_s`, `x_peak_velocity_in_s` | Redundantes — conversão de mm/s |
| `temperature_f` | Redundante — conversão de °C |
| `id`, `created_at` | Vazamento — identificadores correlacionados com a ordem de coleta. Ficam como metadado, não como feature |

Restam **17 features numéricas** para o motor de similaridade.

`rpm` é mantida, mas tratada como **categórica de regime** (5 patamares: 0, 500, 1000,
2000, 3000) — comparar 500 com 3000 como distância contínua não significa nada físico.

### Outliers: identificados, não tratados

Critério de Tukey (IQR), não z-score — várias colunas são fortemente assimétricas
(`z_kurtosis` tem mediana 2,5 e máximo 65), e a média/desvio que o z-score usa já estão
contaminados pelos próprios extremos que deveriam detectar.

**Nada é removido nesta etapa.** Em vibração o pico raro costuma ser o sinal:
kurtosis alta é exatamente a assinatura de impacto de rolamento. Descartar por regra
estatística apagaria a falha que o sistema existe para detectar.

A tabela distingue dois casos:

- `% outliers` alto com `max/limite` ≈ 1 — dispersão larga e uniforme, artefato do
  critério (`z_peak_vel_comp_freq_hz`: o IQR é 2,5 Hz porque a massa está grudada em 61 Hz)
- `% outliers` baixo com `max/limite` nas dezenas — cauda longa de impacto
  (`z_peak_acceleration_g` chega a **49×** o limite superior). **Este é o sinal.**

## Dados e privacidade

O `.gitignore` bloqueia todo insumo da empresa: `data/raw/`, `descricao_desafio/`,
`docs/`, o SQLite gerado e qualquer `.csv`, `.xlsx` ou `.pdf` solto no repositório.
Só `data/fault_map.yaml` é versionado — é decisão curada, não dado.

## Restrições do projeto

- Python, sem dependência de API externa
- Inferência em estação com 32 GB RAM e GPU de 16 GB; LLM local quantizado (7B–8B)
- Banco SQLite (`data/mp.db`), schema em SQLAlchemy — migrar para Postgres é trocar a
  string de conexão

### Fora do MVP

Postgres + pgvector, índice vetorial nativo, classificador supervisionado como sinal
auxiliar de confiança, reranking, autenticação, testes de carga, monitoramento, modelo
maior que 8B.
