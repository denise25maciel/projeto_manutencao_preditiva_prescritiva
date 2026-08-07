# Manutenção Prescritiva em Ativos Rotativos
Denise do Rocio Maciel
Case Técnico — Processo Seletivo 02198/2026 FIESC 
Analista em Pesquisa de Desenvolvimento Tecnológico e Inovação Pleno
Desenvolvedor Full Stack - Pleno - IA e Python
>> Repositório principal: https://github.com/denise25maciel/projeto_manutencao_preditiva_prescritiva.git
>> Repositório segundário (utilizado para aprimorar modelo de ML):https://github.com/denise25maciel/projeto_manutencao_preditiva_prescritiva_classificacao.git


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
