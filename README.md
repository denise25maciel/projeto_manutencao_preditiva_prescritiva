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
| 1 | Eventos e `fault_map.yaml` | ✅ concluída |
| 2 | Banco SQLite via SQLAlchemy | ✅ concluída |
| 3 | Motor de similaridade (kNN) | ⬜ |
| 4 | RAG e guardrails G0–G5 | ⬜ |
| 5 | LLM local (Ollama) | ⬜ |
| 6 | API FastAPI + deploy | ⬜ |
| 7 | Auditoria: assinatura medida × procedimento descrito | ⬜ |

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
| **Análise de Falhas** | Valores únicos de `fault` com busca e filtro. Ao selecionar um rótulo: assinatura de vibração com quartis e CV, o que o distingue do resto do dataset, distribuição de cada feature, **série temporal por coluna** e outliers dentro da classe |
| **Qualidade dos Dados** | Como o dado chegou: nulos por coluna, cadência de coleta, colunas constantes e redundantes, duplicatas e outliers |
| **Documentos** | Os 6 procedimentos convertidos em Markdown: títulos, campos de cada artigo, matriz de campo × arquivo com as pendências, e o diagrama ligando cada procedimento às famílias de `fault` |
| **Eventos** | As 166.796 linhas agrupadas em ocorrências contáveis, com as 5 validações do agrupamento e o custo da regra escolhida |

Na primeira visita à tela **Documentos**, clique em **Converter PDFs** para gerar os
`.md` a partir de `data/raw/*.pdf`. Eles são escritos em
`data/processed/documentos_md/`, fora do git.

O primeiro carregamento lê o CSV (~0,4 s para 166 mil linhas) e guarda em
`@st.cache_data`; os cliques seguintes não releem o arquivo.

> **Ao mexer em `src/mp/`, reinicie o Streamlit.**
>
> O Streamlit recarrega sozinho os arquivos de `ui/`, mas **não** o pacote `mp`,
> que é uma dependência instalada. Se você adicionar uma função em `src/mp/` e o
> app continuar rodando, ele segue com a versão antiga em memória e o sintoma é um
> `ImportError: cannot import name ...` — mesmo com o código correto no disco.
>
> Pare com `Ctrl+C` e rode de novo.

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
│   ├── processed/            # fora do git — .md gerados dos PDFs
│   ├── mp.db                 # fora do git
│   └── fault_map.yaml        # VERSIONADO — é decisão, não dado
├── notebooks/01_eda.ipynb
├── src/mp/
│   ├── config.py             # todo limiar e caminho
│   ├── segmentos.py          # primitiva: agrupar linhas consecutivas
│   ├── analysis/             # DESCREVE: loader, profiling, quality, signatures
│   ├── ingestion/            # TRANSFORMA: sensors (eventos), documents (PDF->MD)
│   └── retrieval/            # catalog: leitura do fault_map.yaml
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

## Documentos de procedimento

Os 6 PDFs de `data/raw/` são convertidos em Markdown com seções numeradas — o
formato que a Parte 4 vai fatiar em chunks. Cada seção é classificada num dos 15
**campos canônicos** (`config.CAMPOS_CANONICOS`), que na Parte 4 viram o metadado
do chunk.

| Arquivo | Título | Seções | Campos |
|---|---|---|---|
| Doc1 | Diagnóstico e Correção de Problemas em **Rolamentos** | 30 | 15/15 |
| Doc2 | Correção de **Desalinhamento** em Motor Elétrico | 24 | 13/15 |
| Doc3 | Correção de **Desbalanceamento** em Máquinas Rotativas | 30 | 13/15 |
| Doc4 | Problemas em Sistemas de Transmissão por **Correias** | 27 | 15/15 |
| Doc5 | Problemas em **Polias** de Sistemas Rotativos | 24 | 14/15 |
| Doc6 | Problemas de Rotor Inclinado (**Cocked Rotor**) | 33 | 15/15 |

### Doc1 veio digitalizado

O PDF de rolamentos são 17 páginas de imagem, sem camada de texto — `pypdf`
extrai 52 caracteres dele, todos de cabeçalho. Rodar OCR exigiria o binário do
Tesseract, que `pip` não resolve e que quebraria o "clone limpo".

O conversor aceita uma **transcrição sidecar** em `data/raw/<nome>.txt` e a usa
quando o PDF não tem texto. A origem fica registrada no front matter de cada `.md`
(`origem_texto: pdf | sidecar`), para ninguém confundir transcrição com extração
automática.

### Campos pendentes

| Documento | Campo ausente |
|---|---|
| Doc2 | Introdução, Indicadores de monitoramento |
| Doc3 | Descrição do problema, Indicadores de monitoramento |
| Doc5 | Descrição do problema |

A ausência de **Indicadores de monitoramento** no Doc2 e no Doc3 é a que importa:
Doc1, Doc4, Doc5 e Doc6 listam quais grandezas acompanhar, e o Doc1 nomeia
`Kurtosis`, `Crest Factor` e `RMS global` — colunas que existem no `banner.csv`.
Sem essa seção, desalinhamento e desbalanceamento não têm ponte explícita entre
procedimento e sensor.

### Cobertura das famílias de `fault`

Dos 16 grupos de `fault`, **9 têm procedimento dedicado**. As famílias de defeito
sem documento — `ventoinha` e `falta_fase`, juntas **13.099 leituras** — são o
caminho de recusa do guardrail **G3**: o fluxo encerra com a mensagem
padronizada, sem chamar o LLM.

`eccentric_rotor` fica como **cobertura parcial**: a seção 3.1 do Doc5 descreve
excentricidade, mas *de polia*, e o rótulo do banner é excentricidade *de rotor*.
Mesmo fenômeno, componente diferente — o G3 **não** libera, porque aceitar a
ligação faria o sistema prescrever ajuste de polia para um problema de rotor.

`normal`, `teste`, `acelerando` e `motor_desligado` também aparecem sem documento,
mas por outro motivo: são **estados**, não defeitos. O **G2** encerra antes do G3.

> **Defeito/estado é decidido pela família, não pelo rótulo solto.** O rótulo
> `new_tes` (2 leituras) é uma truncagem de `new_teste`, e a checagem por
> substring não o reconhece como estado — sozinho, ele fazia a família `teste`
> inteira ser classificada como defeito. A família é a unidade de decisão dos
> guardrails, e é nela que a classificação é aplicada.

## `data/fault_map.yaml` — o catálogo

Único arquivo de `data/` que é **versionado**: é decisão curada, não dado.

Implementa o princípio 1 do projeto. O caminho é sempre

```
rótulo cru  →  família  →  documento
```

e cada seta é um *lookup exato*, nunca uma similaridade. Os 151 rótulos crus,
**incluindo os erros de digitação do operador**, estão listados como `aliases` da
família correta — `cockecocked_adxl_0` resolve para `cocked_rotor`, `new_tes`
para `teste`.

É aqui que os guardrails buscam a resposta:

- **G2** lê `is_problem`. Família com `false` é estado, não defeito — o fluxo
  prescritivo encerra.
- **G3** lê `documentos`. Lista vazia é recusa, **inclusive quando a cobertura é
  `parcial`**. Cobertura parcial não autoriza prescrição.

Leitura pelo módulo [src/mp/retrieval/catalog.py](src/mp/retrieval/catalog.py):

```python
from mp.retrieval import resolver, validar_cobertura

resolver("cocked_rotor_2")
# {'familia': 'cocked_rotor', 'g2_prossegue': True, 'g3_prossegue': True,
#  'documentos': [{'id': 'Doc6', ...}], ...}

resolver("ventoinha")
# g3_prossegue=False — 'Sem documentacao para ventoinha — registre um documento.'
```

`validar_cobertura(df)` confere que todo rótulo do `banner.csv` tem família.
Hoje: **151 de 151, sem órfãos e sem entradas mortas no catálogo.**

`_indice_aliases` levanta erro se o mesmo alias aparecer em duas famílias —
ambiguidade silenciosa no ponto mais crítico do sistema seria pior que uma falha
ruidosa.

## Parte 1 — resultados

**166.796 linhas → 205 eventos.** Um evento é uma vez em que a máquina foi medida
com o mesmo defeito. É o evento que responde *"quantas vezes isso aconteceu"* —
contar linhas responderia 13.000 para `rolamento_inner`, quando foram algumas
medições longas.

A regra quebra **apenas na troca de rótulo**. Cinco verificações binárias confirmam
o agrupamento: nenhum evento mistura rótulos, nenhuma leitura se perdeu ou duplicou,
todas pertencem a um evento, tudo em ordem de data, numeração consecutiva.

### As duas ordens de operação não são equivalentes

| Abordagem | Eventos | Maior duração | Dispersão interna |
|---|---|---|---|
| **A)** ordena por data → separa por rótulo | 205 | 159 h | 2,40 |
| **B)** separa por rótulo → ordena por data | 151 | **943 h** | **3,12** |

Na B, cada grupo tem um rótulo só; o rótulo nunca muda, então nunca há quebra e
cada rótulo vira **um** evento — mesmo tendo sido medido em maio e de novo em junho.
O rótulo `normal` vira um evento de 39 dias.

A **dispersão interna** mede o quanto as leituras de dentro de cada evento se
parecem, com as medidas padronizadas sobre o arquivo inteiro. A B é 30% pior:
ao juntar medições separadas por semanas, ela mistura leituras que não se parecem.

**Usamos a A.** Ela conta ocorrências; a B conta períodos.

### O corte por tempo, adiado mas justificado

A regra atual ignora pausas: se a coleta parou e retomou com o mesmo rótulo, vira um
evento só. Isso acontece em **63 dos 205 eventos**.

Se a decisão mudar, o limiar já está definido e defendido. Os intervalos entre
leituras têm uma **faixa vazia**: nada entre 6,000 s e 16,085 s, em 166.591
intervalos. Qualquer corte nessa faixa dá os mesmos 570 eventos.

Testamos cinco critérios estatísticos automáticos. Tukey e MAD desabam — o IQR vale
0,0003 s porque quase toda leitura tem o mesmo intervalo, e eles devolvem 2 s,
fazendo 26 mil cortes. O que funciona é a **maior descontinuidade relativa**: o
maior salto da distribuição é de 6,000 s para 16,085 s (2,7×), maior que os saltos
entre pausas de horas e de dias. Seu ponto médio dá 11 s.

Registrado em `config.GAP_NOVO_EPISODIO_S = 10 s`, desligado.

## Parte 2 — o banco

Um comando popula tudo:

```bash
python -m mp.db.ingest
```

Gera `data/mp.db` (64 MB) em ~10 s. **Repetível**: cada execução recria o banco do
zero, então rodar duas vezes dá exatamente o mesmo resultado.

### Quatro tabelas

| Tabela | Linhas | O que responde |
|---|---|---|
| `readings` | 166.796 | como a máquina vibrou naquele instante |
| `episodes` | 356 | quantas vezes isso aconteceu e quando |
| `documents` | 6 | existe procedimento para essa falha |
| `chunks` | 168 | qual trecho do procedimento responde |

`readings` tem **duas** colunas de evento — `evento_a` e `evento_b` — porque as duas
ordens de operação estão guardadas lado a lado. `episodes` usa a chave composta
**(versao, numero)**: 205 eventos da versão A mais 151 da B.

O caminho de uma consulta:

```
leitura → evento → rótulo → família → documento → trecho
```

Cada seta é uma consulta exata. A ligação **família → documento** é a única que não
está no banco: vive no `fault_map.yaml`, versionado no Git, porque é decisão curada
— no Git cada mudança tem autor, data e motivo; no banco viraria um `UPDATE` sem rastro.

### Verificação

18 checagens conferem o banco contra o CSV: contagens, ausência de nulos, valores de
200 leituras sorteadas, soma das leituras por versão, coerência entre `evento_a`/
`evento_b` e o rótulo, e integridade referencial dos chunks.

### Uma limitação do SQLite a saber

O esquema declara `DateTime(timezone=True)`, mas **o SQLite não armazena fuso**. As
datas voltam sem `+00:00` — o instante está certo, só a etiqueta se perde. Todo
`created_at` é UTC por construção. Comparar com uma data com fuso levanta
`TypeError`; use `.replace(tzinfo=timezone.utc)` no que veio do banco.

Some sozinho na migração para PostgreSQL.

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
