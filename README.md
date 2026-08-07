# Manutenção Preditiva e Prescritiva

Sistema de manutenção prescritiva para máquinas rotativas de chão de fábrica.

O técnico descreve o problema — ou cola um trecho de leituras do sensor de
vibração. O sistema identifica o defeito, encontra o procedimento correspondente
e explica o que fazer **citando documento, seção e página**. Se não houver
procedimento para aquela falha, ele diz isso, em vez de improvisar.

```
"a bomba está com vibração alta no mancal e um ruído grave"
            ↓
    16 famílias de falha, 6 procedimentos, 168 trechos indexados
            ↓
"Trata-se de desalinhamento. Segundo o Doc2, seção 19 (página 4),
 afrouxe os parafusos da base e meça o desvio radial com relógio
 comparador antes de calçar."
```

A ideia central: **o modelo de linguagem só redige.** Ele não decide qual é a
falha, não escolhe o manual e não inventa número. Tudo isso é decidido por
código — consultas exatas e verificações determinísticas — antes de qualquer
palavra chegar ao modelo. Uma falha sem procedimento é recusada **sem que uma
única linha seja enviada ao LLM**.

---

## Índice

- [O que o sistema faz](#o-que-o-sistema-faz)
- [Estado atual](#estado-atual)
- [Documentação](#documentação)
- [▶ Como executar — passo a passo](#-como-executar--passo-a-passo)
- [Como usar a interface](#como-usar-a-interface)
- [Arquitetura](#arquitetura)
- [Resultados por etapa](#resultados-por-etapa)
- [Dados e privacidade](#dados-e-privacidade)

---

## O que o sistema faz

Existem **duas fontes** de informação, e elas nunca se misturam:

| Fonte | O que é | Como é usada |
|---|---|---|
| `banner.csv` | 166.796 leituras de sensor de vibração, com a falha anotada pelo operador | treina um classificador que nomeia a família da falha |
| 6 PDFs | procedimentos de manutenção escritos por especialistas | viram 168 trechos buscáveis por significado |

A única ponte entre as duas é a coluna `fault`, através de um catálogo curado à
mão (`data/fault_map.yaml`). **Número nunca é comparado com texto.** O trecho de
sensor resolve para uma família; a família resolve para um documento por consulta
exata. Isso não é detalhe de implementação — é o que impede o sistema de
prescrever "ajuste de polia" para um problema de rotor só porque os dois textos
se parecem.

### O caminho de uma pergunta

```
1. o que é essa falha?        ← floresta de 400 árvores (números, não texto)
2. é defeito ou é estado?     ← SELECT no catálogo
3. existe manual para ela?    ← SELECT no catálogo   ⟵ pode recusar aqui
4. trava o manual da conversa ← irreversível
5. a pergunta cabe no manual? ← código               ⟵ pode recusar aqui
6. busca os trechos           ← filtro exato + cosseno
7. os trechos servem?         ← código               ⟵ pode recusar aqui
8. REDIGIR                    ← ⭐ o único ponto com LLM
9. as citações existem?       ← código               ⟵ pode descartar o texto
```

Há **nove maneiras de o fluxo parar. Em oito delas o modelo nem é chamado.**

---

## Estado atual

| Parte | Escopo | Status |
|---|---|---|
| 0 | Módulo de análise exploratória | ✅ concluída |
| 1 | Eventos e `fault_map.yaml` | ✅ concluída |
| 2 | Banco SQLite via SQLAlchemy | ✅ concluída |
| 3 | ~~Motor de similaridade (kNN)~~ — **removido** | ➖ substituído |
| — | Classificação supervisionada (Random Forest) | ✅ concluída |
| 4 | RAG e guardrails | ✅ concluída |
| 5 | O agente: grafo, multi-turno, cliente plugável | ✅ concluída |
| 6 | API FastAPI + Docker | ⬜ não iniciada |
| 7 | Auditoria: assinatura medida × procedimento descrito | ⬜ não iniciada |

**Duas ressalvas honestas:**

> **A Parte 3 mudou de rota.** O plano previa kNN por similaridade. Na prática
> entrou uma **floresta aleatória** treinada sobre janelas de leituras, vinda de
> um repositório irmão sobre os mesmos dados. Ela nomeia a família que abre a
> conversa, mas **não escolhe manual e não entra no caminho do LLM** — com 44% de
> acurácia honesta, o veredito dela passa antes pelo guardrail G1.

> **A Parte 6 não existe ainda.** Não há `api/` nem `docker-compose.yml`. A
> interface Streamlit importa `src/mp/` diretamente, através de um único arquivo
> ponte (`ui/_dados.py`). Funciona, e o caching do Streamlit evita recarregar os
> modelos — mas não é o desenho final descrito no plano.

---

## Documentação

Este README cobre **o que é** e **como rodar**. O detalhamento vive em `docs/`:

| Arquivo | Responde |
|---|---|
| `ARQUITETURA.md` | o sistema explicado do zero — os cinco princípios, cada módulo, um exemplo completo ponta a ponta, glossário |
| `COMUNICACOES.md` | quem conversa com quem: onde o código toca o banco, onde toca a rede, o mapa de chamadas |
| `LIMPEZA.md` | o que é limpo antes de entrar no modelo e no chat, e o que **não** é limpo — com o motivo de cada decisão |

> ⚠️ **A pasta `docs/` não é versionada.** O `.gitignore` bloqueia a pasta
> inteira, porque ela também recebe insumos da empresa. Num clone limpo esses
> arquivos não vêm junto — peça-os a quem mantém o projeto.

Além deles, o código é a documentação mais confiável: cada decisão numérica vive
em [`src/mp/config.py`](src/mp/config.py) com o motivo em comentário, e os
módulos explicam nas docstrings **por que** fazem o que fazem, não só o quê.

A própria interface é documentação: a página **Contextualização** percorre os
dados do geral ao específico, mostrando cada achado com o número que o sustenta.

---

## ▶ Como executar — passo a passo

### Antes de começar

| Requisito | Detalhe |
|---|---|
| **Python 3.11 ou superior** | validado em 3.14.4, Windows 11. Confira com `python --version` |
| **Git** | para clonar |
| **Espaço em disco** | ~500 MB na instalação mínima; **~3 GB** se instalar a busca neural (que traz o PyTorch) |
| **Os insumos** | `banner.csv` e os 6 PDFs de procedimento — **não estão no repositório** |
| **Chave de LLM** | *opcional*. Sem ela o sistema roda e devolve o texto cru do manual |

> **Nada de GPU é necessário.** A floresta roda em CPU e o embedder é pequeno. A
> GPU só entra se você optar por rodar um LLM local pesado.

---

### Roteiro rápido

Para quem já conhece o terreno. Cada linha está explicada em detalhe abaixo.

```bash
git clone <url-do-repositorio>
cd projeto_manutencao_preditiva_prescritiva

python -m venv .venv
source .venv/Scripts/activate          # Windows Git Bash

pip install --upgrade pip
pip install -e ".[ml]"                 # mínimo que funciona

mkdir -p data/raw
cp /caminho/banner.csv data/raw/
cp /caminho/*.pdf      data/raw/

streamlit run ui/Contextualizacao.py   # → converta os PDFs no bloco 4, depois Ctrl+C

python -m mp.db.ingest                                       # cria data/mp.db
python -c "import sys; sys.path.insert(0,'src'); from mp.retrieval import rag; rag.indexar()"

streamlit run ui/Contextualizacao.py   # agora vale tudo
```

---

### Passo 1 — Clonar o repositório

```bash
git clone <url-do-repositorio>
cd projeto_manutencao_preditiva_prescritiva
```

Confira que você está na raiz certa — todos os comandos seguintes partem daqui:

```bash
ls          # deve mostrar: README.md  pyproject.toml  src/  ui/  data/
```

---

### Passo 2 — Criar e ativar o ambiente virtual

O `venv` isola as dependências do projeto do Python do sistema. A pasta `.venv/`
está no `.gitignore` — cada máquina cria a sua.

**Windows — PowerShell**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows — Git Bash**
```bash
python -m venv .venv
source .venv/Scripts/activate
```

**Linux / macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

> Se o PowerShell recusar com `execution of scripts is disabled on this system`,
> rode **uma vez**:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

**Como saber se deu certo:** o prompt fica prefixado com `(.venv)`. Para
confirmar sem depender do visual:

```bash
python -c "import sys; print(sys.prefix)"
# deve terminar em .../.venv  — se apontar para o Python do sistema, o venv não está ativo
```

> ⚠️ **O venv precisa estar ativo em toda sessão de terminal nova.** Se você
> fechar o terminal e voltar amanhã, ative de novo antes de qualquer comando.

---

### Passo 3 — Instalar as dependências

As dependências são **em camadas**. Instale só o que for usar.

#### 3a. O mínimo que faz tudo funcionar

```bash
python -m pip install --upgrade pip
pip install -e ".[ml]"
```

Isso instala o pacote em modo editável (`-e`), o que resolve o `import mp` de
qualquer lugar, mais:

| Pacote | Para quê |
|---|---|
| `pandas`, `numpy`, `pyarrow` | manipulação dos dados |
| `streamlit`, `altair` | a interface e os gráficos |
| `pyyaml` | ler o `fault_map.yaml` |
| `pypdf` | extrair texto dos procedimentos |
| `sqlalchemy` | o banco |
| `langchain-core`, `pydantic` | mensagens tipadas — **não fala com provedor nenhum** |
| `python-dotenv`, `httpx` | ler o `.env`, chamar o Ollama |
| `scikit-learn` (extra `ml`) | a floresta **e** a busca por TF-IDF |

**Com isto o sistema já roda inteiro**: análise, classificação, busca nos
manuais, guardrails e conversa. O que falta é qualidade de busca e a redação em
prosa.

#### 3b. Busca por significado (opcional, pesado)

```bash
pip install -e ".[ml,rag]"
```

Traz o `sentence-transformers` e um modelo multilíngue — melhor para documentos
em português. **Custa ~2,5 GB**, porque puxa o PyTorch.

Sem ele, a busca cai automaticamente para TF-IDF + LSA. **Funciona igual, muda a
qualidade** — nada quebra.

> ⚠️ **Decida isto antes do Passo 7.** Os vetores gravados no banco carregam o
> nome do modelo que os gerou, e vetores de modelos diferentes não são
> comparáveis. Se você indexar com TF-IDF e depois instalar o
> `sentence-transformers`, a busca avisa *"reindexe"* e para. Não é um bug —
> é a trava funcionando.

#### 3c. Modelo de linguagem (opcional)

Instale **apenas o provedor que for usar**:

```bash
pip install langchain-ollama       # local, sem chave — é a meta do projeto
pip install langchain-openai       # exige OPENAI_API_KEY
pip install langchain-anthropic    # exige ANTHROPIC_API_KEY
```

**Sem nenhum deles o sistema continua funcionando.** A conversa devolve o texto
cru do manual, com documento, seção e página. E isso é de propósito: é a prova
visível de que o conteúdo não vem do modelo.

#### Sobre o `requirements.txt`

O arquivo existe e funciona, mas é um `pip freeze` do ambiente completo de
desenvolvimento — **173 pacotes**, incluindo Jupyter, matplotlib e PyTorch.

Duas coisas a saber antes de usá-lo:

1. Ele contém uma linha `-e git+https://github.com/...`, que faz o pip **clonar o
   projeto de novo** durante a instalação.
2. Está codificado em UTF-16, não em UTF-8. O pip lê, mas outras ferramentas
   podem não ler.

**Prefira o `pip install -e ".[ml]"` da seção 3a.** Use o `requirements.txt` só
se precisar reproduzir o ambiente exato, e nesse caso remova a linha `-e git+`
antes.

---

### Passo 4 — Colocar os insumos no lugar

Os dados são da empresa e **não estão no repositório** — o `.gitignore` bloqueia
`data/raw/`, todo `.csv` e todo `.pdf`.

```bash
mkdir -p data/raw
cp /caminho/para/banner.csv data/raw/
cp /caminho/para/*.pdf      data/raw/
```

Ao final, `data/raw/` deve conter:

```
data/raw/
├── banner.csv          ← 166.796 linhas
├── <rolamentos>.pdf
├── <desalinhamento>.pdf
├── <desbalanceamento>.pdf
├── <correias>.pdf
├── <polias>.pdf
└── <cocked_rotor>.pdf
```

**Alternativa sem mover o arquivo** — aponte uma variável de ambiente:

```bash
export MP_CSV=/caminho/para/banner.csv          # Linux / macOS / Git Bash
```
```powershell
$env:MP_CSV = "C:\caminho\para\banner.csv"      # PowerShell
```

O [`config.py`](src/mp/config.py) procura nesta ordem: `MP_CSV` → `data/raw/` →
`docs/` → `descricao_desafio/` → `teste_industria/`. Se não achar, a interface
mostra uma mensagem dizendo onde colocar, em vez de quebrar.

> **Um dos PDFs veio digitalizado** (o de rolamentos): são 17 páginas de imagem,
> sem camada de texto. Rodar OCR exigiria o binário do Tesseract, que o `pip` não
> resolve. A saída foi aceitar uma **transcrição** em `data/raw/<mesmo-nome>.txt`,
> que o conversor usa quando o PDF não tem texto. A origem fica registrada em
> cada `.md` (`origem_texto: pdf | sidecar`), para ninguém confundir transcrição
> com extração automática.

---

### Passo 5 — Configurar a chave do LLM *(pule se não for usar)*

Crie um arquivo `.env` **na raiz do projeto**:

```ini
# .env — nunca vai para o Git
OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# OLLAMA_URL=http://localhost:11434
```

**Para rodar local com Ollama** (a meta do projeto — sem chave, sem API externa):

```bash
# 1. instale o Ollama: https://ollama.com/download
# 2. baixe um modelo de 7B–8B
ollama pull llama3.1:8b
# 3. confira que o serviço responde
curl http://localhost:11434/api/tags
```

Modelos que a interface oferece: `llama3.1:8b`, `qwen2.5:7b`, `mistral:7b`,
`gemma2:9b`. Ela lista **os que estiverem realmente baixados** na máquina.

> O `.env` está no `.gitignore`. Nunca coloque chave no código.

---

### Passo 6 — Converter os PDFs em Markdown

Os procedimentos precisam virar `.md` com seções numeradas antes de irem para o
banco. **Isso é feito pela interface**, não há comando de terminal:

```bash
streamlit run ui/Contextualizacao.py
```

Na página que abrir, vá até o **bloco 4 — Os documentos** e clique em
**Converter PDFs**.

Os arquivos são escritos em `data/processed/documentos_md/` (fora do Git). Cada
seção sai marcada com o seu tipo:

```markdown
#### 19. Correção do Desalinhamento <!-- campo: correcao | pagina: 4 -->
```

Esse rótulo é **regex no título, não LLM** — é ele que permite pedir "o que
fazer" sem saber de antemão em que seção isso está.

Confira que deu certo:

```bash
ls data/processed/documentos_md/     # devem aparecer 6 arquivos .md
```

Depois **pare o Streamlit** com `Ctrl+C`.

---

### Passo 7 — Popular o banco

```bash
python -m mp.db.ingest
```

Saída esperada:

```
1/5  criando o esquema do zero...
2/5  lendo o banner.csv...
     166.796 leituras
3/5  montando os eventos nas duas versoes...
     versao A: 526 eventos | versao B: 436 eventos
4/5  gravando leituras e eventos...
5/5  gravando documentos e secoes...

pronto em 16.0s — 64.0 MB
  leituras      166.796
  eventos           962
  documentos          6
  chunks            168
  banco        data/mp.db
```

**É repetível de propósito:** cada execução apaga e recria o banco. Rodar duas
vezes produz exatamente o mesmo resultado — o que permite ajustar uma regra de
agrupamento e regerar sem medo de duplicar.

> Se aparecer `nenhum .md encontrado — rode a conversao dos PDFs antes`, volte ao
> Passo 6. O banco é criado assim mesmo, mas sem documento nenhum, e a conversa
> não terá o que citar.

---

### Passo 8 — Indexar os trechos ⚠️

**Este passo é obrigatório e não tem botão.** Sem ele, a busca nos manuais
responde *"Os trechos ainda não foram indexados"* e a conversa não sai do lugar.

```bash
python -c "import sys; sys.path.insert(0,'src'); from mp.retrieval import rag; rag.indexar()"
```

Saída esperada:

```
168 trechos indexados com paraphrase-multilingual-MiniLM-L12-v2 (384 dimensoes)
```

ou, sem o `sentence-transformers` instalado:

```
168 trechos indexados com tfidf-lsa (128 dimensoes)
```

São 168 trechos: segundos com TF-IDF, um pouco mais com o modelo neural — que na
primeira vez também **baixa os pesos** (~120 MB).

> **Quando reindexar:** sempre que reconverter os PDFs, rodar a ingestão de novo,
> ou trocar de embedder (instalar/desinstalar o `sentence-transformers`).

---

### Passo 9 — Abrir a interface

```bash
streamlit run ui/Contextualizacao.py
```

Abre em **http://localhost:8501**.

Se a porta estiver ocupada:

```bash
streamlit run ui/Contextualizacao.py --server.port 8502
```

---

### Verificação final

Percorra esta lista. Se algo falhar, a coluna da direita diz para onde voltar.

| # | Confira | Se falhar |
|---|---|---|
| 1 | `python -c "import sys; print(sys.prefix)"` termina em `.venv` | Passo 2 |
| 2 | `python -c "import mp; print('ok')"` imprime `ok` | Passo 3 |
| 3 | `ls data/raw/banner.csv` existe | Passo 4 |
| 4 | `ls data/processed/documentos_md/` tem 6 arquivos | Passo 6 |
| 5 | `ls -lh data/mp.db` mostra ~64 MB | Passo 7 |
| 6 | a Contextualização abre e mostra 166.796 leituras | Passos 4 e 9 |
| 7 | em **Diagnóstico**, descrever um problema traz um manual e trechos | Passo 8 |
| 8 | na aba **Modelo de linguagem**, "Testar conexão" responde | Passos 3c e 5 |

Os itens 1 a 7 **não dependem de LLM nenhum**. Se todos passarem, o sistema está
funcionando — o item 8 só acrescenta a redação em prosa.

---

### Problemas comuns

| Sintoma | Causa | Solução |
|---|---|---|
| `ModuleNotFoundError: No module named 'mp'` | pacote não instalado, ou venv inativo | ative o venv e rode `pip install -e ".[ml]"` |
| `ModuleNotFoundError: No module named 'sklearn'` | faltou o extra `ml` | `pip install -e ".[ml]"` — a interface não abre sem ele |
| `banner.csv nao encontrado. Procurei em: ...` | insumo fora do lugar | Passo 4, ou defina `MP_CSV` |
| `Os trechos ainda nao foram indexados. Rode indexar()` | faltou o Passo 8 | rode o comando de indexação |
| `Os trechos foram indexados com 'X' e a busca esta usando 'Y'` | trocou de embedder depois de indexar | rode o Passo 8 de novo |
| `OPENAI_API_KEY nao encontrada` | sem `.env` | Passo 5 — ou use o sistema sem LLM, que funciona |
| A conversa devolve texto do manual, sem prosa | nenhum provedor instalado ou selecionado | Passo 3c — **e isto é o comportamento correto**, não um erro |
| `execution of scripts is disabled` | política do PowerShell | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `ImportError: cannot import name ...` depois de editar `src/mp/` | Streamlit não recarrega o pacote `mp` | `Ctrl+C` e rode de novo |
| Porta 8501 ocupada | outra instância aberta | `--server.port 8502` |

> **Ao mexer em `src/mp/`, reinicie o Streamlit.** Ele recarrega sozinho os
> arquivos de `ui/`, mas **não** o pacote `mp`. O app segue com a versão antiga em
> memória, e o sintoma é um `ImportError` mesmo com o código correto no disco.

---

### Rodar sem interface

O pipeline de classificação roda inteiro no terminal e imprime o mesmo relatório
da aba 3 — as cinco etapas cronometradas, as métricas, os folds:

```bash
python -m mp.classificacao.execucao
```

E os módulos podem ser usados direto:

```python
from mp.analysis import carregar, perfil_rotulos, assinaturas_por_rotulo
from mp.retrieval import verificar_existencia_conserto

df = carregar()
perfil_rotulos(df)                              # rótulos, contagem, janela temporal
assinaturas_por_rotulo(df, min_leituras=100)    # tabela de assinaturas

c = verificar_existencia_conserto("cocked_rotor_2")
c.familia, c.situacao, c.prescrever              # ('cocked_rotor', 'ok', True)
c.documento_ids                                  # ['Doc6']
```

---

## Como usar a interface

### Contextualização — a página inicial

Uma sequência única, do geral ao específico, em quatro blocos:

| Bloco | O que responde |
|---|---|
| **1. Os dados gerais** | quanto dado, de quando, em que ritmo — e os números que provam que o arquivo não está em ordem de data |
| **2. As colunas** | o que cada medida é, quais duas medem a mesma grandeza em unidades diferentes, onde há valor fora do normal, quais podem sair |
| **3. As falhas** | que defeitos existem, como cada um se comporta, e quantas vezes cada um aconteceu **de verdade** |
| **4. Os documentos** | os 6 procedimentos em Markdown, a matriz de campo × arquivo, e o diagrama ligando procedimento a família |

A ordem carrega o argumento. Primeiro o **arquivo** — quanto veio, se dá para
confiar no horário. Depois as **colunas**, porque não dá para ler a assinatura de
uma falha sem saber que duas delas medem a mesma coisa. Só então as **falhas**. E
por fim os **documentos**, que são a outra fonte.

### Diagnóstico — onde o sistema é usado

| Aba | O que faz |
|---|---|
| **Diagnóstico e conversa** | o fluxo completo. O técnico descreve o problema e, se tiver, cola um **trecho** de leituras em CSV no mesmo formulário. Com o trecho, a classificação entra como a **primeira fala** da conversa — os números da floresta, redigidos pelo modelo e conferidos pelo guardrail G5N. Depois vem o procedimento, citando documento, seção e página |
| **Modelo de linguagem** | quais provedores estão disponíveis, a configuração, as regras do prompt e o teste de conexão — com uma conversa livre, **sem guardrail nenhum**, de propósito, para servir de contraste |

A configuração era tela separada e virou aba: trocar de modelo no meio de uma
conversa não pode custar sair da tela e perder o fio.

### Classificação — a pergunta anterior a todas

*Dá para descobrir a família só pelos números do sensor, sem ninguém anotar?*

| Aba | O que faz |
|---|---|
| **Preparação dos dados** | como uma leitura vira um exemplo, em 6 passos — até a transformação de cada janela em 80 números, mostrada lado a lado com as 50 leituras cruas |
| **O modelo e o que ele vale** | a floresta de 400 árvores, as duas maneiras de cortar treino e teste, onde o modelo erra, e um evento segurado fora do treino para experimentar |
| **Executar e ver o resultado** | o pipeline rodando de verdade: 5 etapas cronometradas, e no fim o laudo — métricas, folds, matriz de confusão |

As duas primeiras leem de cache; a terceira **não usa cache nenhum**, de
propósito — na segunda execução os tempos apareceriam próximos de zero, e um
painel de execução que não mede execução é enfeite.

---

## Arquitetura

### Os cinco princípios

1. **A coluna `fault` é a chave de junção.** Números nunca são comparados com
   texto. O evento resolve para um rótulo; o rótulo resolve para um documento via
   `data/fault_map.yaml`, curado à mão e versionado.
2. **Fronteira determinística/generativa.** Toda pergunta respondível por consulta
   ao banco não passa pelo LLM. O modelo recebe números prontos; nunca os produz.
3. **Guardrails são código, não prompt.** Oito verificações determinísticas: G0 a
   G5, mais a G1T (a evidência aponta um manual?) e a G5N (os números da prosa
   foram apurados?).
4. **Busca vetorial nunca retorna vazio** — por isso o G3 é um `SELECT` no
   catálogo, nunca uma similaridade semântica. Só uma consulta exata pode devolver
   lista vazia, e lista vazia é a única forma honesta de dizer "não sei".
5. **Sem duplicação de lógica.** A interface importa `src/mp/`. Nunca reimplementa.

### Os guardrails

| ID | Verifica | Se falhar |
|----|----------|-----------|
| G0 | schema e faixas físicas da entrada | rejeita |
| G1 | confiança da classificação ≥ 0,30 | "sem base para nomear a falha" |
| G1T | a evidência separa os manuais candidatos | devolve a lista para o técnico escolher |
| G2 | o rótulo é problema, não estado | encerra — não há o que corrigir |
| G3 | a falha tem documento no catálogo | "sem documentação — registre um documento" |
| G4 | os trechos passam do score mínimo | trata como G3 |
| G5 | as citações da resposta existem nos trechos enviados | regenera 2×, depois mostra o trecho cru |
| G5N | os números da prosa foram apurados | regenera 2×, depois a frase é escrita por código |

### Onde o código toca o mundo

**Um arquivo fala com o banco** — [`retrieval/rag.py`](src/mp/retrieval/rag.py)
lê; [`db/ingest.py`](src/mp/db/ingest.py) escreve. Nenhum outro módulo abre sessão
de banco.

**Um arquivo fala com a rede** — [`llm/client.py`](src/mp/llm/client.py). Não há
outra chamada externa em todo o `src/mp/`.

Todo o resto é função pura recebendo e devolvendo dado.

### A interface não faz parte do runtime

`ui/` **documenta decisões** — o porquê de cada limiar, cada descarte de coluna,
cada escolha de agregação. Ela chama `src/mp/` e não reimplementa nada, e é por
isso que remover a interface inteira não quebraria o pipeline:
`python -m mp.classificacao.execucao` chama os mesmos módulos.

### Estrutura

```
├── data/
│   ├── raw/                    # fora do git — banner.csv e os PDFs
│   ├── processed/              # fora do git — os .md gerados
│   ├── mp.db                   # fora do git — gerado pela ingestão
│   └── fault_map.yaml          # VERSIONADO — é decisão, não dado
├── src/mp/
│   ├── config.py               # todo limiar e caminho, com o motivo
│   ├── segmentos.py            # primitiva: agrupar linhas consecutivas
│   ├── analysis/               # DESCREVE: loader, profiling, quality, signatures
│   ├── ingestion/              # TRANSFORMA: sensors (eventos), documents (PDF→MD)
│   ├── classificacao/          # amostras, modelo (RandomForest), validacao
│   ├── db/                     # models, session, ingest
│   ├── retrieval/              # catalog (fault_map), rag (busca), embeddings
│   ├── guardrails/             # rules.py — as oito travas
│   ├── llm/                    # client (3 provedores), prompts
│   ├── agente/                 # estado.py (Sessao/Turno), grafo.py (os nós)
│   └── pipeline.py             # a versão de turno único
├── ui/
│   ├── Contextualizacao.py     # entrypoint: a narrativa dos dados
│   ├── _dados.py               # ponte cacheada UI → mp
│   ├── _secao_*.py             # blocos com render()
│   └── pages/                  # Diagnóstico, Classificação
├── requirements.txt            # freeze do ambiente de desenvolvimento
└── pyproject.toml              # dependências e extras — prefira este
```

---

## Resultados por etapa

### Parte 0 — análise

166.796 leituras × 26 colunas, coletadas entre 30/04 e 16/06/2026.

**Três achados que contrariam a suposição inicial:**

**1. `created_at` não está em ordem cronológica.** Há saltos negativos de dezenas
de dias entre linhas vizinhas: são ~331 sessões gravadas em épocas diferentes e
concatenadas fora de ordem. *Consequência:* toda operação que depende de
vizinhança temporal precisa ordenar antes.

**2. `z_peak_vel_comp_freq_hz` e `x_peak_vel_comp_freq_hz` não são constantes em
61 Hz.** Têm 79 e 50 valores distintos; 61 Hz é a moda (60% e 49% das linhas), não
o valor único. As colunas carregam informação e **não devem ser descartadas** — a
frequência do pico se desloca justamente em alguns defeitos.

**3. São 151 rótulos distintos, não ~10.** A inflação vem de erros de digitação
(`mortor_desligado_novo`, `normla_carga_3_3`, `cockecocked_adxl_0`), sufixos de
sessão e do prefixo `new_`. Consolidam em **16 famílias**, sem sobra.

**Confirmado:** zero nulos nas 26 colunas; cadência de ~2 s (92% dentro de
±0,25 s); unidades duplicadas (`mm/s = in/s × 25,4`, `°F = °C × 9/5 + 32`),
confirmadas por identidade numérica, não por correlação; **5,8% de duplicatas
consecutivas** — 9.736 linhas idênticas à anterior.

**Colunas descartadas** (para o modelo, não para o armazenamento):

| Coluna | Motivo |
|---|---|
| 4 × `*_in_s`, `temperature_f` | redundantes — conversão de unidade |
| `id`, `created_at` | vazamento — correlacionados com a ordem de coleta |
| `rpm`, `temperature_c` | regime, não sintoma |

Restam **16 colunas de medida**.

**Outliers: identificados, não tratados.** Critério de Tukey (IQR), não z-score —
várias colunas são fortemente assimétricas, e a média/desvio que o z-score usa já
estão contaminados pelos extremos que deveriam detectar. **Nada é removido:** em
vibração o pico raro costuma ser o sinal. `z_peak_acceleration_g` chega a **49×**
o limite superior — este é o sinal, não o ruído.

### Parte 1 — eventos e catálogo

**166.796 linhas → 526 eventos.** Um evento é uma vez em que a máquina foi medida
com o mesmo defeito, na mesma rotação. É o evento que responde *"quantas vezes
isso aconteceu"* — contar linhas responderia 13.000 para `rolamento_inner`, quando
foram algumas medições longas.

**Por que a rotação encerra um evento.** A primeira versão quebrava só na troca de
rótulo, e **136 dos 205 eventos misturavam rotações** — 95% das leituras. A
bancada rodava 500, 1000 e 2000 rpm sem trocar o nome da falha. Num caso, a
velocidade RMS ia de 3,5 a **21,1 mm/s** dentro do "mesmo" evento.

| Regra | Eventos | Dispersão interna | Com rotação misturada |
|---|---|---|---|
| Só rótulo | 205 | 2,40 | **136** |
| **Rótulo + rotação** | **526** | **1,31** | **0** |

**As duas ordens de operação não são equivalentes:**

| Abordagem | Eventos | Maior duração | Dispersão |
|---|---|---|---|
| **A)** ordena → separa | 526 | 101 h | **1,31** |
| **B)** separa → ordena | 436 | 943 h | 1,36 |

Na B, o rótulo `normal` vira **um** evento de 39 dias. **Usamos a A** — ela conta
ocorrências; a B conta períodos.

### O catálogo `fault_map.yaml`

Único arquivo de `data/` que é versionado: é decisão curada, não dado. O caminho é
sempre `rótulo cru → família → documento`, e cada seta é um *lookup exato*.

Os 151 rótulos crus, **incluindo os erros de digitação do operador**, estão
listados como aliases da família correta. `verificar_existencia_conserto`
classifica em quatro situações:

| Situação | Exemplo | O que o técnico vê | G2 | G3 |
|---|---|---|:--:|:--:|
| `ok` | `cocked_rotor` | o procedimento | ✅ | ✅ |
| `estado` | `normal`, `teste` | "está operando, não há o que corrigir" | ❌ | — |
| `sem_documento` | `ventoinha` | "é defeito, mas falta o manual — registre" | ✅ | ❌ |
| `desconhecido` | fora do catálogo | "condição nova, ninguém registrou" | ❌ | — |

Dizer "sem documentação" quando a máquina está apenas normal seria mentir — por
isso as quatro são nomeadas separadamente.

**Cobertura:** 151 de 151 rótulos, sem órfãos. Dos 16 grupos, **9 têm procedimento
dedicado**. `ventoinha` e `falta_fase` (13.099 leituras) são o caminho de recusa
do G3. `eccentric_rotor` fica como cobertura **parcial**: o Doc5 descreve
excentricidade *de polia*, e o rótulo é excentricidade *de rotor* — mesmo
fenômeno, componente diferente, e aceitar a ligação faria o sistema prescrever
ajuste de polia para um problema de rotor.

### Parte 2 — o banco

| Tabela | Linhas | O que responde |
|---|---|---|
| `readings` | 166.796 | como a máquina vibrou naquele instante |
| `episodes` | 962 | quantas vezes isso aconteceu e quando |
| `documents` | 6 | existe procedimento para essa falha |
| `chunks` | 168 | qual trecho do procedimento responde |

`readings` tem **duas** colunas de evento porque as duas ordens de operação estão
guardadas lado a lado. 18 checagens conferem o banco contra o CSV.

> **Limitação do SQLite:** o esquema declara `DateTime(timezone=True)`, mas o
> SQLite **não armazena fuso**. As datas voltam sem `+00:00` — o instante está
> certo, a etiqueta se perde. Tudo é UTC por construção. Some sozinho na migração
> para PostgreSQL.

### Documentos de procedimento

| Arquivo | Título | Seções | Campos |
|---|---|---|---|
| Doc1 | **Rolamentos** | 30 | 15/15 |
| Doc2 | **Desalinhamento** em motor elétrico | 24 | 13/15 |
| Doc3 | **Desbalanceamento** | 30 | 13/15 |
| Doc4 | **Correias** | 27 | 15/15 |
| Doc5 | **Polias** | 24 | 14/15 |
| Doc6 | **Cocked rotor** | 33 | 15/15 |

**Campos pendentes** — a ausência de *Indicadores de monitoramento* no Doc2 e no
Doc3 é a que importa: os outros quatro listam quais grandezas acompanhar, e o Doc1
nomeia `Kurtosis`, `Crest Factor` e `RMS global`, colunas que existem no
`banner.csv`. Sem essa seção, desalinhamento e desbalanceamento não têm ponte
explícita entre procedimento e sensor.

**O ponto fraco conhecido:** se um documento novo chamar a seção de "Ações
Recomendadas" e nenhum padrão de título pegar, o campo fica `NULL` e aquela seção
**some da busca prescritiva**, que filtra por tipo. Falha em silêncio.
`campos_pendentes` é a checagem que pega isso.

---

## Pendências em aberto

Decisões adiadas de propósito, documentadas para não virarem esquecimento.

| ID | Pendência | Situação |
|---|---|---|
| **P1** | 1.000 leituras de `rolamento_outer_2` com o mesmo carimbo de tempo | ficam como estão — as medidas variam, então o dado é real; só o carimbo está errado. Corrigir por estimativa inventaria dado |
| **P2** | 9.736 duplicatas consecutivas (5,84%) | detectadas e exibidas, **não removidas**. Como a janela anda de 25 em 25, um trecho duplicado gera janelas quase idênticas que pesam no treino como evidência independente |
| **P3** | descarte de colunas | **aplicado** para o modelo, em `classificacao/colunas.py`; o banco segue guardando tudo |

---

## Dados e privacidade

O `.gitignore` bloqueia todo insumo da empresa: `data/raw/`, `descricao_desafio/`,
`docs/`, o SQLite gerado e qualquer `.csv`, `.xlsx` ou `.pdf` solto no
repositório. Só `data/fault_map.yaml` é versionado — é decisão curada, não dado.

Chaves de API vivem no `.env`, que também está bloqueado. Nunca no código.

---

## Restrições do projeto

- Python, **sem dependência de API externa** na entrega
- Inferência em estação com 32 GB RAM e GPU de 16 GB; LLM local quantizado (7B–8B)
- Banco **SQLite** (`data/mp.db`), schema em SQLAlchemy — migrar para PostgreSQL é
  trocar a string de conexão, numa função só

> **Por que SQLite e não Postgres.** Não é limitação técnica: é para o avaliador
> conseguir verificar. O banco é um arquivo único, que ele abre com qualquer
> visualizador e confere contra o CSV. Um serviço para subir seria uma barreira
> entre ele e a checagem.

**Fora do MVP:** Postgres + pgvector, índice vetorial nativo, reranking,
autenticação, testes de carga, monitoramento, modelo maior que 8B.
