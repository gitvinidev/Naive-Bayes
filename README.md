# Classificador Bayesiano — Predição de Defeitos de Software

Atividade Prática 1 da disciplina de Mineração de Dados: implementação de um
classificador **Naive Bayes em SQL**, do zero (priors, verossimilhanças com
suavização de Laplace, classificação por log-probabilidades), aplicado a um
domínio real de engenharia de software.

| | |
|---|---|
| **Autor** | Marcus Viníicius Santos de Almeida |
| **Disciplina** | Mineração de Dados |
| **Entrega** | 04/09/2026 |
| **Banco de dados** | SQLite (sem dependências externas) |

## O problema

Rodar a suíte de testes completa a cada alteração e revisar manualmente todo o
código não escala em projetos grandes. Este projeto modela a **priorização**
de teste/revisão como classificação binária: dado um arquivo de código-fonte
descrito por métricas estáticas e de histórico, prever se ele é propenso a
apresentar um defeito reportado (`SIM` / `NÃO`).

A saída é um sinal de priorização — não um veredito. Um arquivo classificado
como `SIM` entra na frente da fila de revisão.

## O pipeline, em 4 etapas

```
Etapa 1                Etapa 2                 Etapa 3                  Etapa 4
Modelagem       ──▶     Dados            ──▶    Classificador     ──▶   Resultados
domínio, rótulo,        gera 150 módulos        views SQL: priors,      6 casos de teste,
6 features,              sintéticos,             verossimilhanças        log-odds por
discretização            independentes            (Laplace), score        feature, reflexão
                         entre si                 em log, classificação   crítica
     │                        │                          │                     │
     ▼                        ▼                          ▼                     ▼
etapa1_modelagem.pdf   gerar_dados.py            classificador_...sql   rodar_casos_teste.py
                        → *.csv / *.txt           rodar_classificador.py  → resultados_casos.txt
```

Cada seta é um artefato executável; rodando os 3 scripts em sequência (ver
[Como executar](#como-executar)) o pipeline completo é reconstruído do zero.

---

## Etapa 1 — Modelagem do problema

**Rótulo alvo:** o arquivo apresentará um defeito reportado na janela de
observação? `SIM` (propenso) / `NÃO` (não propenso) — variável binária que o
Naive Bayes estima como `P(SIM | features)`.

**Unidade de análise:** arquivo de código-fonte (não classe, como no estudo de
referência de Koru et al.) — simplificação declarada, já que nenhuma das 6
features depende do conceito de classe.

**As 6 features:**

| # | Feature | Coleta (em um cenário real) | Faixas (`baixo` · `medio` · `alto`) |
|---|---|---|---|
| 1 | Complexidade ciclomática | `radon cc` | ≤10 · 11–20 · >20 |
| 2 | Linhas de código (LOC) | `radon raw` | <50 · 50–200 · >200 |
| 3 | Nº de autores distintos | `git log` / PyDriller | 1 · 2–3 · ≥4 |
| 4 | Churn relativo (90 dias) | PyDriller | <10% · 10–30% · >30% |
| 5 | Nº de imports/dependências | módulo `ast` (stdlib) | 0–3 · 4–8 · >8 |
| 6 | Cobertura de testes | Coverage.py | <50% · 50–80% · >80% |

Cada feature tem fundamentação própria em literatura de engenharia de software
empírica (McCabe, Shepperd, Koru et al., Nagappan & Ball, Matsumoto et al.,
Inozemtseva & Holmes...) — ver
[`relatorios/etapa1_modelagem.pdf`](relatorios/etapa1_modelagem.pdf) para a
justificativa completa, a lógica de discretização e as referências em ABNT.

---

## Etapa 2 — Massa de dados sintética

`dados/gerar_dados.py` gera **150 módulos** sintéticos (mínimo exigido: 100).
Decisão de projeto — **"Naive Bayes puro"** (ver `CLAUDE.md`): as 6 features
são sorteadas de forma **independente entre si** (cada uma com seu próprio
gerador de números aleatórios, via `rng.spawn()`); só o **rótulo** depende das
6 features, o que é esperado — é literalmente o que o Naive Bayes modela.

Isso testa o classificador no cenário em que sua própria suposição central
(independência condicional) é verdadeira **por construção**, isolando o
comportamento teórico do algoritmo — não é uma afirmação de que features como
complexidade e LOC seriam de fato independentes em código real (não são; ver
[Limitações](#limitações-conhecidas)).

**Matriz de correlação de Pearson resultante** (`dados/etapa2_validacao.txt`)
— todos os pares saem com \|r\| < 0,10, confirmando a independência:

| | Complex. | LOC | Autores | Churn | Imports | Cobert. |
|---|---|---|---|---|---|---|
| **Complex.** | 1,00 | 0,01 | 0,00 | 0,07 | 0,08 | 0,00 |
| **LOC** | 0,01 | 1,00 | −0,04 | −0,06 | 0,04 | 0,06 |
| **Autores** | 0,00 | −0,04 | 1,00 | −0,07 | −0,09 | −0,04 |
| **Churn** | 0,07 | −0,06 | −0,07 | 1,00 | −0,01 | 0,01 |
| **Imports** | 0,08 | 0,04 | −0,09 | −0,01 | 1,00 | −0,04 |
| **Cobert.** | 0,00 | 0,06 | −0,04 | 0,01 | −0,04 | 1,00 |

Proporção de classes: **34,7 % SIM / 65,3 % NÃO** (~1:2, moderadamente
desbalanceado de propósito). Detalhes do modelo generativo, estatísticas
descritivas e a checagem de que a taxa de defeito ainda é monotônica por
feature isolada em
[`relatorios/etapa2_dados.pdf`](relatorios/etapa2_dados.pdf).

---

## Etapa 3 — Classificador Naive Bayes em SQL

`sql/classificador_naive_bayes.sql` implementa o classificador como uma
cadeia de `VIEW`s em SQLite:

`priors` (P(classe)) → `treino_longo` (unpivot das 6 categorias) →
`verossimilhancas` (P(categoria\|classe) com suavização de Laplace) →
`score_log` (soma de log-probabilidades) → `classificar_modulo`
(normalização de volta para % + recomendação).

**Em uma frase:** para cada classe, soma-se o logaritmo do prior com o
logaritmo da verossimilhança (Laplace) de cada uma das 6 features — evitando
o *underflow* de multiplicar probabilidades pequenas diretamente — e
normaliza-se o resultado de volta para uma probabilidade entre 0% e 100%.

`sql/rodar_classificador.py` (re)constrói `dados/classificador.db`, importa o
CSV da Etapa 2 e roda um *smoke test* com 2 casos de exemplo. Priors:
**P(NÃO) = 98/150 = 0,653** · **P(SIM) = 52/150 = 0,347**. Arquitetura
completa, view a view, em
[`relatorios/etapa3_classificador.pdf`](relatorios/etapa3_classificador.pdf).

---

## Etapa 4 — Resultados dos testes e reflexão crítica

`testes/rodar_casos_teste.py` classifica **6 casos formais**
(`testes/casos_teste.csv`) e calcula o log-odds de cada categoria de cada
feature a partir da view `verossimilhancas`.

| Caso | Perfil | P(SIM) | Recomendação |
|---|---|---|---|
| a — baixo risco claro | tudo bom | 4,28% | BAIXO RISCO |
| b — alto risco claro | tudo ruim | 95,53% | ALTO RISCO |
| c — ambíguo | complexidade alta × cobertura alta | 38,04% | BAIXO RISCO (fronteira) |
| d — "armadilha" de Koru et al., sem bônus de interação | arquivo pequeno e denso | 55,52% | ALTO RISCO (por pouco) |
| e — combinação rara | grande, simples, churn alto | 22,14% | BAIXO RISCO |
| f — feature contestada | tudo ruim + cobertura alta | 92,64% | ALTO RISCO |

**Poder discriminativo das features** (ranking por \|log-odds\| médio):
nº de autores (0,695) > complexidade ciclomática (0,595) > churn relativo
(0,482) > LOC (0,285) > cobertura de testes (0,227) > nº de imports (0,180).

**Achado central — caso d.** Sem nenhum bônus de interação plantado nos
dados (Etapa 2), o resultado do arquivo "pequeno e denso" é exatamente a
**soma** dos dois efeitos marginais — log-odds de LOC baixo (+0,45) mais
log-odds de complexidade alta (+0,76) — e nada além disso. O Naive Bayes soma
evidências marginais; ele não modela, para mais nem para menos, um efeito de
interação entre features.

Análise completa dos 6 casos, decomposição em log-odds e a Reflexão Crítica
(Seção 5) em
[`relatorios/etapa4_resultados.pdf`](relatorios/etapa4_resultados.pdf).

---

## Limitações conhecidas

- **A independência entre features testada aqui é verdadeira por construção,
  não por realismo.** Os dados sintéticos são gerados com as 6 features
  independentes entre si — a matriz de correlação confirma \|r\| < 0,10 em
  todos os pares. Isso valida a implementação do classificador no cenário
  ideal, mas não reflete um cenário real de implantação: em código real,
  complexidade ciclomática e LOC são fortemente correlacionadas (R² ≈ 0,93 —
  Shepperd, 1988), e um Naive Bayes treinado sobre dados reais contaria essa
  evidência de tamanho/estrutura praticamente duas vezes. É uma limitação
  **teórica, documentada na literatura**, não uma violação encontrada nestes
  dados.
- **Relação não-linear de LOC com risco.** Arquivos pequenos são
  proporcionalmente mais propensos a defeito (Koru et al., 2008) — "menor"
  não é sinônimo de "mais seguro". Essa relação é de cada feature isolada e
  continua presente mesmo com os dados independentes.
- **Cobertura de testes é uma feature de evidência contestada** — mantida de
  propósito para ser analisada criticamente, não por forte poder preditivo.
- **Módulo = arquivo, não classe.** Simplificação deliberada frente ao estudo
  de referência, que usa classe como unidade.
- **Dados de treinamento sintéticos**, com o rótulo gerado a partir de pesos
  por categoria de cada feature (exigência da atividade).

Discussão completa de cada ponto no relatório da Etapa 4.

---

## Como executar

Requer apenas Python 3 com `numpy`, `pandas` (geração dos dados) e `sqlite3`
(biblioteca padrão, classificador). `markdown` + `weasyprint` são necessários
apenas para regenerar os PDFs (`relatorios/build_pdf.py`), não para rodar o
classificador.

```bash
# 1. Gerar a massa de dados sintética de treinamento (150 registros)
python3 dados/gerar_dados.py

# 2. Construir o classificador e rodar o smoke test
python3 sql/rodar_classificador.py

# 3. Rodar os 6 casos de teste formais e calcular os log-odds
python3 testes/rodar_casos_teste.py
```

Os três scripts são idempotentes — podem ser executados quantas vezes for
preciso, sempre com o mesmo resultado (semente fixa `SEED = 42`).

## Estrutura do repositório

```
.
├── CLAUDE.md                          # contexto do projeto para agentes de IA
├── relatorios/                        # Etapa 1–4: .md fonte + .pdf de cada relatório
│   ├── etapa1_modelagem.{md,pdf}      # domínio, rótulo, features, discretização
│   ├── etapa2_dados.{md,pdf}          # metodologia da massa de dados sintética
│   ├── etapa3_classificador.{md,pdf}  # arquitetura do SQL
│   ├── etapa4_resultados.{md,pdf}     # casos de teste, log-odds, reflexão crítica
│   └── build_pdf.py                   # .md -> HTML com estilo -> PDF (WeasyPrint)
├── dados/                              # Etapa 2
│   ├── gerar_dados.py                 # gera a massa sintética de treinamento
│   ├── etapa2_dados_treinamento.csv   # 150 registros
│   └── etapa2_validacao.txt           # matriz de correlação, estatísticas
├── sql/                                # Etapa 3
│   ├── classificador_naive_bayes.sql  # priors, verossimilhanças, classificação
│   └── rodar_classificador.py         # importa o CSV, roda o SQL, smoke test
└── testes/                             # Etapa 4
    ├── casos_teste.csv                # 6 perfis de teste
    ├── rodar_casos_teste.py           # roda os casos e calcula log-odds
    └── resultados_casos.txt
```

## Referências principais

- MCCABE, T. J. *A Complexity Measure*. IEEE TSE, 1976.
- SHEPPERD, M. *A Critique of Cyclomatic Complexity as a Software Metric*. Software Engineering Journal, 1988.
- KORU, A. G. et al. *Theory of Relative Defect Proneness*. Empirical Software Engineering, 2008.
- NAGAPPAN, N.; BALL, T. *Use of Relative Code Churn Measures to Predict System Defect Density*. ICSE, 2005.
- MATSUMOTO, S. et al. *An Analysis of Developer Metrics for Fault Prediction*. PROMISE, 2010.
- WEYUKER, E. J.; OSTRAND, T. J.; BELL, R. M. *Do Too Many Cooks Spoil the Broth?* Empirical Software Engineering, 2008.
- INOZEMTSEVA, L.; HOLMES, R. *Coverage Is Not Strongly Correlated with Test Suite Effectiveness*. ICSE, 2014.
- GREN, L.; ANTINYAN, V. *On the Relation Between Unit Testing and Code Quality*. SEAA, 2017.

Lista completa em formato ABNT no relatório da Etapa 1.

## Sobre o uso de IA

Domínio, features, dados sintéticos, código SQL e análise foram construídos em
diálogo com uma IA generativa, usada como parceira de modelagem. Cada
alegação teórica trazida pela IA foi verificada contra a literatura original
antes de ser incorporada — o processo de checagem está documentado no
relatório da Etapa 1 (Apêndice B). Todo o conteúdo é defensável oralmente,
linha a linha, pelo autor.
