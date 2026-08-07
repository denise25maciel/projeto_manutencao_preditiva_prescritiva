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
| 3 | ~~Motor de similaridade (kNN)~~ — **removido** | ➖ |
| 4 | RAG e guardrails | ✅ concluída |
| 5 | O agente: grafo, multi-turno, cliente plugável | ✅ concluída |
| — | Classificação supervisionada (Random Forest) | ✅ concluída |
| 6 | API FastAPI + deploy | ⬜ |
| 7 | Auditoria: assinatura medida × procedimento descrito | ⬜ |

> **Sobre a Parte 5.** O fluxo está completo e o cliente é plugável desde o
> primeiro dia, mas o desenvolvimento usou a API da OpenAI. **Trocar para o
> Ollama não muda uma linha do pipeline** — é uma escolha na interface. A meta
> de entrega continua sendo local.
>
> A **classificação supervisionada** não estava no plano original: veio de um
> repositório irmão, sobre os mesmos dados, e entrou como sinal auxiliar. Ela
> não decide manual nem entra no caminho do modelo de linguagem.

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
streamlit, altair, pyyaml e scikit-learn.
O `scikit-learn` deixou de ser opcional: a interface não abre sem ele — a
floresta de classificação depende dele. As bibliotecas que
ainda faltam (FastAPI, `sentence-transformers`) estão listadas em comentário no
fim do arquivo e mapeadas como extras no `pyproject.toml`.

**Modelo de linguagem (Parte 5).** O `langchain-core` já está na lista, mas ele
não fala com provedor nenhum — traz só as mensagens tipadas e a saída
estruturada. O adaptador do provedor é um pacote separado, e você instala apenas
o que for usar:

```bash
pip install langchain-ollama      # a meta: roda local, sem chave
pip install langchain-openai      # para desenvolver, exige OPENAI_API_KEY no .env
```

Sem nenhum deles o sistema continua funcionando: a conversa devolve o texto cru
do manual, e é assim que se comprova que o conteúdo não vem do modelo.

> O import de `src/mp/` funciona sem instalar o pacote — `ui/_dados.py` resolve
> o caminho sozinho. Se preferir o pacote instalado, use `pip install -e .`
> (opcional).

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
streamlit run ui/Contextualizacao.py
```

Abre em `http://localhost:8501`. A **Contextualização** é a página inicial: uma
sequência única, do geral ao específico, em quatro blocos.

| Bloco | O que responde |
|---|---|
| **1. Os dados gerais** | Quanto dado, de quando, em que ritmo. O arquivo cru, sem tratamento, e os números que provam que ele não está em ordem de data |
| **2. As colunas** | O que cada medida é: quantos valores distintos tem, quais duas medem a mesma grandeza em unidades diferentes, onde há valor fora do normal, quais podem sair |
| **3. As falhas** | Que defeitos existem, como cada um se comporta, e quantas vezes cada um aconteceu de verdade — as 166.796 linhas viram ocorrências contáveis |
| **4. Os documentos** | Os 6 procedimentos convertidos em Markdown, a matriz de campo × arquivo com as pendências, e o diagrama ligando cada procedimento às famílias de `fault` |

A ordem carrega o argumento. Primeiro o **arquivo** — quanto veio, de quando, se
dá para confiar no horário. Depois as **colunas**, porque não dá para ler a
assinatura de uma falha sem saber que duas delas medem a mesma coisa. Só então as
**falhas**. E por fim os **documentos**, que são a outra fonte — e as duas só se
encontram pela coluna `fault`, nunca por semelhança entre número e texto.

No menu lateral fica a tela que **usa** isso — o **Diagnóstico**, em duas abas:

| Aba | O que faz |
|---|---|
| **Diagnóstico e conversa** | O fluxo completo: o técnico descreve o problema e, se tiver, cola um **trecho** de leituras do sensor em CSV no mesmo formulário. Com o trecho, a **classificação entra na conversa como a primeira fala** — os números da floresta, redigidos pelo modelo e conferidos pelo G5N. Depois vem o procedimento, citando documento, seção e página |
| **Modelo de linguagem** | Quais provedores estão disponíveis, a configuração, as regras do prompt e o teste de conexão — com uma conversa livre, sem guardrail nenhum, de propósito, para servir de contraste |

A configuração era uma tela separada e virou aba (`ui/_secao_modelo.py`): trocar
de modelo no meio de uma conversa não pode custar sair da tela e perder o fio.

E a tela **Classificação**, que responde a pergunta anterior a todas as outras —
*dá para descobrir a família só pelos números do sensor, sem ninguém anotar?*
Em três abas:

| Aba | O que faz |
|---|---|
| **Preparação dos dados** | Como uma leitura vira um exemplo, em 6 passos: o rótulo resolvido pelo `fault_map.yaml`, o agrupamento em eventos, as colunas escolhidas, o recorte em janelas de 50 leituras, a transformação de cada janela em 80 números — mostrada lado a lado, as 50 leituras cruas contra as estatísticas que saem delas — e o corte nas **duas bases**, treino e teste, ambas visíveis e baixáveis |
| **O modelo e o que ele vale** | A floresta de 400 árvores, as duas maneiras de cortar treino e teste, onde o modelo erra, um evento segurado fora do treino para experimentar, e os dois experimentos de configuração |
| **Executar e ver o resultado** | O pipeline rodando de verdade: as 5 etapas em ordem, cada uma cronometrada e mostrando o que produziu, e no fim o laudo dos testes — métricas, folds, matriz de confusão e relatório para baixar |

As duas primeiras leem de cache; a terceira **não usa cache nenhum**, de
propósito — na segunda execução os tempos apareceriam próximos de zero, e um
painel de execução que não mede execução é enfeite. O mesmo relatório sai no
terminal, sem Streamlit:

```bash
python -m mp.classificacao.execucao
```

Ela vem de um repositório irmão de classificação, sobre os mesmos dados, e foi
**adaptada, não copiada**: o algoritmo veio inteiro, mas cada *decisão* passou a
sair de onde este projeto já a tomava — o rótulo do `fault_map.yaml` em vez de
regras no código, o grupo do evento (`fault` + `rpm`) em vez da troca de rótulo,
e as colunas de um lugar só, `classificacao/colunas.py`. O detalhe está em
`src/mp/classificacao/`.

Os quatro blocos moravam em cinco `ui/_secao_*.py`; hoje são funções dentro do
próprio `ui/Contextualizacao.py`. Continuam funções, e não código solto no nível
do módulo, porque `filtro`, `colunas` e `total` são nomes que quase todos usam —
no mesmo escopo, colidiriam em silêncio.

Na primeira visita, no **ato 6**, clique em **Converter PDFs** para gerar os
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

### Sem abrir a interface

O pipeline de classificação roda inteiro no terminal e imprime o mesmo relatório
que a aba 3 mostra — as cinco etapas cronometradas, as métricas, os folds:

```bash
python -m mp.classificacao.execucao
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
3. **Guardrails são código, não prompt.** Oito verificações determinísticas, na
   ordem: G0 a G5, mais a G1T (a evidência aponta um manual?) e a G5N (os números
   da prosa foram apurados?).
4. **Busca vetorial nunca retorna vazio** — por isso G3 é um `SELECT` no catálogo, nunca
   uma similaridade semântica.
5. **Sem duplicação de lógica.** A UI importa `src/mp/`. Nunca reimplementa.

### A interface não faz parte do runtime

`ui/` **documenta decisões** — o porquê de cada limiar, cada descarte de coluna,
cada escolha de agregação. Ela chama `src/mp/` e não reimplementa nada, e é por
isso que remover a interface inteira não quebraria o pipeline: a API e
`python -m mp.classificacao.execucao` chamam os mesmos módulos.

### Estrutura

```
├── data/
│   ├── raw/                  # fora do git — dado da empresa
│   ├── processed/            # fora do git — .md gerados dos PDFs
│   ├── mp.db                 # fora do git
│   └── fault_map.yaml        # VERSIONADO — é decisão, não dado
├── src/mp/
│   ├── config.py             # todo limiar e caminho
│   ├── segmentos.py          # primitiva: agrupar linhas consecutivas
│   ├── analysis/             # DESCREVE: loader, profiling, quality, signatures
│   ├── ingestion/            # TRANSFORMA: sensors (eventos), documents (PDF->MD)
│   ├── classificacao/        # amostras, modelo (RandomForest), validacao
│   └── retrieval/            # catalog: leitura do fault_map.yaml
├── ui/
│   ├── Contextualizacao.py   # entrypoint: a narrativa dos dados, em 4 blocos
│   ├── _dados.py             # ponte cacheada UI -> mp.analysis
│   ├── _secao_*.py           # blocos com render(): o modelo e a classificação
│   └── pages/                # só o que é tela de verdade: diagnóstico, classificação
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

Restam **16 colunas de medida** para a floresta (`classificacao/colunas.py`).

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

### O `campo` — como o sistema sabe qual seção é a correção

Cada seção numerada é classificada em um **campo canônico** — `sintomas`,
`diagnostico`, `correcao`, `validacao`, `seguranca`… — e é esse rótulo que
permite pedir "o que fazer" sem saber de antemão em que seção isso está.

**A classificação é regex no título, não LLM.**
[`classificar_campo`](src/mp/ingestion/documents.py#L141) compara o título com os
padrões de `config.CAMPOS_CANONICOS` na conversão do PDF para `.md`. Uma seção
chamada "19. Correção do Desalinhamento" vira `campo: correcao` porque a palavra
está no título — nenhum modelo leu o conteúdo para concluir isso. Título que não
casa com nenhum padrão fica `NULL`.

A marca fica visível no `.md`, como comentário HTML:

```markdown
#### 19. Correção do Desalinhamento <!-- campo: correcao | pagina: 4 -->
```

A ingestão lê esse comentário de volta e grava na coluna `chunks.campo`
([models.py:238](src/mp/db/models.py#L238)), com índice composto
`(documento_id, campo)` — exatamente os dois filtros que a busca aplica antes de
tocar em vetor.

**É assim que a busca prescritiva funciona**
([`buscar_prescritivo`](src/mp/retrieval/rag.py#L322)), em três passos:

1. `SELECT` pelo documento — a família dá o manual, e só os trechos dele entram
2. filtro por tipo — ficam os de `correcao`, `validacao` e `criterios_aceitacao`
3. cosseno — entre esses, a pergunta do técnico ordena por semelhança

O sistema **não sabe** que a correção do Doc1 está nas seções 19 a 22. Ele sabe
que existem trechos do tipo `correcao` e deixa a semelhança decidir quais deles
respondem àquela pergunta. Nada aqui consulta o `fault_map.yaml` — o campo
`secoes_correcao` que existe lá **não é lido por nenhuma linha de código**;
sobrou da curadoria manual da Parte 1 e a classificação por título o tornou
desnecessário.

**O ponto fraco, e por isso a seção seguinte existe:** se um documento novo
chamar a seção de "Ações Recomendadas" e nenhum padrão pegar, o `campo` fica
`NULL` e aquela seção **some da busca prescritiva** — que filtra por tipo. Falha
em silêncio, sem erro e sem aviso.

### Campos pendentes

`campos_pendentes` ([documents.py:416](src/mp/ingestion/documents.py#L416)) é a
checagem que pega isso: lista cada documento sem algum campo canônico.

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

É aqui que os guardrails buscam a resposta. Quem decide é
[`verificar_existencia_conserto`](src/mp/retrieval/catalog.py), o ponto de
entrada único do catálogo:
G2 e G3 leem o veredito dele em vez de reabrir o YAML cada um por sua conta.

Ele classifica o rótulo em uma de **quatro situações**. Do ponto de vista do
fluxo as três últimas terminam igual — sem prescrição, sem chamar o modelo —,
mas são notícias diferentes para quem lê:

| `situacao` | Exemplo | O que o técnico vê | G2 | G3 |
|---|---|---|:--:|:--:|
| `ok` | `cocked_rotor` | o procedimento | ✅ | ✅ |
| `estado` | `normal`, `teste` | "está operando, não há o que corrigir" | ❌ | — |
| `sem_documento` | `ventoinha`, `eccentric_rotor` | "é defeito, mas falta o manual — registre" | ✅ | ❌ |
| `desconhecido` | rótulo fora do catálogo | "condição nova, ninguém registrou" | ❌ | — |

Dizer "sem documentação" quando a máquina está apenas normal seria mentir, e não
há nada a registrar — por isso as quatro são nomeadas separadamente.

`eccentric_rotor` cai em `sem_documento` mesmo aparecendo no Doc5: cobertura
**parcial** conta como ausência. É excentricidade de polia, não de rotor.

```python
from mp.retrieval import verificar_existencia_conserto, validar_cobertura

c = verificar_existencia_conserto("cocked_rotor_2")
c.familia, c.situacao, c.prescrever   # ('cocked_rotor', 'ok', True)
c.documento_ids                       # ['Doc6']

c = verificar_existencia_conserto("ventoinha")
c.e_defeito, c.prescrever             # (True, False)  — G2 passa, G3 recusa
c.mensagem  # "Sem documentacao para 'ventoinha' — registre um documento."
```

`e_defeito` e `prescrever` são leituras de `situacao`, não campos independentes:
não há como os dois vereditos divergirem entre si.

`validar_cobertura(df)` confere que todo rótulo do `banner.csv` tem família.
Hoje: **151 de 151, sem órfãos e sem entradas mortas no catálogo.**

`_indice_aliases` levanta erro se o mesmo alias aparecer em duas famílias —
ambiguidade silenciosa no ponto mais crítico do sistema seria pior que uma falha
ruidosa.

## Parte 1 — resultados

**166.796 linhas → 526 eventos.** Um evento é uma vez em que a máquina foi medida
com o mesmo defeito, na mesma rotação. É o evento que responde *"quantas vezes isso
aconteceu"* — contar linhas responderia 13.000 para `rolamento_inner`, quando foram
algumas medições longas.

A regra quebra na troca de **rótulo** ou de **rotação**. Seis verificações binárias
confirmam o agrupamento: nenhum evento mistura rótulos, nenhum mistura rotações,
nenhuma leitura se perdeu ou duplicou, todas pertencem a um evento, tudo em ordem de
data, numeração consecutiva.

### Por que a rotação encerra um evento

A primeira versão quebrava só na troca de rótulo. O resultado: **136 dos 205 eventos
misturavam rotações** — 95% das leituras. A bancada rodava 500, 1000 e 2000 rpm em
sequência sem trocar o nome da falha, então três ensaios viravam um evento só.

No caso extremo, um evento de `rolamento_combination_pos_2` tinha velocidade RMS de
3,5 mm/s a 500 rpm e **21,1 a 2000 rpm** — seis vezes maior dentro do "mesmo" evento.
A mediana dele não descrevia nenhum dos três regimes.

| Regra | Eventos | Dispersão interna | Com rotação misturada |
|---|---|---|---|
| Só rótulo | 205 | 2,40 | **136** |
| **Rótulo + rotação** | **526** | **1,31** | **0** |

### As duas ordens de operação não são equivalentes

| Abordagem | Eventos | Maior duração | Dispersão interna |
|---|---|---|---|
| **A)** ordena por data → separa | 526 | 101 h | 1,31 |
| **B)** separa → ordena por data | 436 | **943 h** | 1,36 |

Na B, cada grupo tem um rótulo só; o rótulo nunca muda, então nunca há quebra e
cada rótulo vira **um** evento — mesmo tendo sido medido em maio e de novo em junho.
O rótulo `normal` vira um evento de 39 dias.

A **dispersão interna** mede o quanto as leituras de dentro de cada evento se
parecem, com as medidas padronizadas sobre o arquivo inteiro. A B é 30% pior:
ao juntar medições separadas por semanas, ela mistura leituras que não se parecem.

**Usamos a A.** Ela conta ocorrências; a B conta períodos.

### O corte por tempo, adiado mas justificado

A regra atual ignora pausas: se a coleta parou e retomou com o mesmo rótulo, vira um
evento só. Isso acontece em **28 dos 526 eventos**.

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

Gera `data/mp.db` (64 MB) em ~16 s. **Repetível**: cada execução recria o banco do
zero, então rodar duas vezes dá exatamente o mesmo resultado.

### Quatro tabelas

| Tabela | Linhas | O que responde |
|---|---|---|
| `readings` | 166.796 | como a máquina vibrou naquele instante |
| `episodes` | 962 | quantas vezes isso aconteceu e quando |
| `documents` | 6 | existe procedimento para essa falha |
| `chunks` | 168 | qual trecho do procedimento responde |

`readings` tem **duas** colunas de evento — `evento_a` e `evento_b` — porque as duas
ordens de operação estão guardadas lado a lado. `episodes` usa a chave composta
**(versao, numero)**: 526 eventos da versão A mais 436 da B.

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
