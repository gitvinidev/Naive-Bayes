# Classificador Bayesiano — Predição de Defeitos de Software

Atividade Prática 1 da disciplina de Mineração de Dados: implementação de um
classificador **Naive Bayes em SQL**, do zero (priors, verossimilhanças com
suavização de Laplace, classificação por log-probabilidades), aplicado a um
domínio real de engenharia de software.

| | |
|---|---|
| **Autor** | Marcus Vinícius Santos de Almeida |
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

## Features

| Feature | Coleta (em um cenário real) | Faixas |
|---|---|---|
| Complexidade ciclomática | `radon cc` | baixa ≤10 · média 11–20 · alta >20 |
| Linhas de código (LOC) | `radon raw` | curto <50 · médio 50–200 · longo >200 |
| Nº de autores distintos | `git log` / PyDriller | 1 · 2–3 · ≥4 |
| Churn relativo (90 dias) | PyDriller | <10% · 10–30% · >30% |
| Nº de imports/dependências | módulo `ast` (stdlib) | 0–3 · 4–8 · >8 |
| Cobertura de testes | Coverage.py | <50% · 50–80% · >80% |

Cada feature tem fundamentação em literatura de engenharia de software
empírica — ver [`relatorios/etapa1_modelagem.pdf`](relatorios/etapa1_modelagem.pdf)
para a justificativa completa e as referências.

## Estrutura do repositório

```
.
├── CLAUDE.md                          # contexto do projeto para agentes de IA
├── relatorios/
│   ├── etapa1_modelagem.pdf           # domínio, features, discretização
│   ├── etapa2_dados.pdf               # metodologia da massa de dados sintética
│   ├── etapa3_classificador.pdf       # arquitetura do SQL
│   └── etapa4_resultados.pdf          # casos de teste, log-odds, reflexão crítica
├── dados/
│   ├── gerar_dados.py                 # gera a massa sintética de treinamento
│   ├── etapa2_dados_treinamento.csv   # 150 registros
│   └── etapa2_validacao.txt           # matriz de correlação, checagem dos padrões
├── sql/
│   ├── classificador_naive_bayes.sql  # priors, verossimilhanças, classificação
│   └── rodar_classificador.py         # importa o CSV, roda o SQL, smoke test
└── testes/
    ├── casos_teste.csv                # 6 perfis de teste
    ├── rodar_casos_teste.py           # roda os casos e calcula log-odds
    └── resultados_casos.txt
```

## Como executar

Requer apenas Python 3 (`sqlite3` já vem na biblioteca padrão — nenhuma
dependência externa é necessária para os artefatos principais).

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

## Como o classificador funciona, em uma frase

Para cada classe, soma-se o logaritmo da probabilidade a priori com o
logaritmo da verossimilhança (com suavização de Laplace) de cada uma das 6
features observadas — evitando o *underflow* numérico de multiplicar
probabilidades pequenas diretamente — e depois normaliza-se o resultado de
volta para uma probabilidade entre 0% e 100%. Detalhes de cada view SQL estão
em [`relatorios/etapa3_classificador.pdf`](relatorios/etapa3_classificador.pdf).

## Resultados

| Caso | Perfil | P(SIM) | Recomendação |
|---|---|---|---|
| a — baixo risco claro | tudo bom | 4,17% | BAIXO RISCO |
| b — alto risco claro | tudo ruim | 95,61% | ALTO RISCO |
| c — ambíguo | complexidade alta × cobertura alta | 61,13% | ALTO RISCO (fronteira) |
| d — "armadilha" de Koru et al. | arquivo pequeno e denso | 74,40% | ALTO RISCO |
| e — combinação rara | grande, simples, churn alto | 17,12% | BAIXO RISCO |
| f — feature contestada | tudo ruim + cobertura alta | 96,29% | ALTO RISCO |

**Poder discriminativo das features** (ranking por \|log-odds\| médio):
complexidade ciclomática (0,857) > nº de imports (0,467) > nº de autores
(0,386) > LOC (0,328) > churn relativo (0,292) > cobertura de testes (0,167).

Análise completa dos 6 casos, decomposição em log-odds e reflexão crítica em
[`relatorios/etapa4_resultados.pdf`](relatorios/etapa4_resultados.pdf).

## Limitações conhecidas

- **Violação de independência (complexidade × LOC).** As duas features têm
  correlação empírica alta (R² ≈ 0,93 na literatura; 0,67 de Pearson nos
  dados deste projeto) — o modelo conta a evidência de tamanho/estrutura
  praticamente duas vezes.
- **Relação não-linear de LOC com risco.** Arquivos pequenos são
  proporcionalmente mais propensos a defeito (Koru et al., 2008) — "menor"
  não é sinônimo de "mais seguro".
- **Cobertura de testes é uma feature de evidência contestada** — mantida de
  propósito para ser analisada criticamente, não por forte poder preditivo.
- **Módulo = arquivo, não classe.** Simplificação deliberada frente ao estudo
  de referência, que usa classe como unidade.
- **Dados de treinamento sintéticos**, gerados por script com padrões
  programados a partir da literatura (exigência da atividade).

Discussão completa de cada ponto no relatório da Etapa 4.

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

