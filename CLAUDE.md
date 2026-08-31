# Contexto do Projeto — Atividade Prática 1: Classificador Bayesiano

## Sobre a atividade
Implementação de um Classificador Naive Bayes em SQL, aplicado a um domínio real,
com dados de treinamento, casos de teste e reflexão crítica. Trabalho individual.
Entrega: até 04/09/2026.

Entregáveis exigidos (5 no total):
1. Relatório de modelagem (Etapa 1)
2. Massa de dados — 100+ registros (Etapa 2)
3. Código do classificador em SQL, comentado (Etapa 3)
4. Resultados de 5+ casos de teste com análise (Etapa 4)
5. Reflexão crítica (mínimo 1 parágrafo — pode compor o relatório da Etapa 4)

Critérios de avaliação: coerência da modelagem (25%), corretude do algoritmo (30%),
qualidade dos dados (15%), análise crítica (20%), clareza da documentação (10%).

Este é um trabalho individual — todo conteúdo gerado deve ser algo que o autor
consiga explicar oralmente, linha por linha (exigência explícita da atividade).
Evitar complexidade desnecessária ou trechos que não sejam justificáveis numa
arguição.

## Domínio escolhido
**Predição de Defeitos de Software a partir de métricas estáticas de código.**

Rótulo alvo: o módulo (arquivo) de código vai apresentar um defeito reportado (SIM/NÃO).

Justificativa: domínio de mineração de dados/engenharia de software com
literatura acadêmica consolidada, dados fáceis de obter com poucas ferramentas,
e utilidade prática real — priorizar teste e revisão de código, já que rodar a
suíte de testes completa ou revisar manualmente todo o código não escala em
projetos grandes.

## Unidade de análise ("módulo") = arquivo
O estudo de referência (Koru et al.) usa **classe** como módulo. Aqui adotamos
**arquivo** como simplificação: nenhuma das 6 features atuais depende
estritamente do conceito de classe (as duas que dependiam, DIT e acoplamento,
foram removidas — ver histórico abaixo), então arquivo funciona como unidade
universal, sem exigir ferramentas adicionais. Declarar essa simplificação no
relatório.

## As 6 features

| # | Feature | Ferramenta | Fundamentação e pontos de atenção |
|---|---------|------------|-------------------------------------|
| 1 | Complexidade ciclomática | Radon (`radon cc`) | McCabe (1976) criou a métrica para apoiar desenho de casos de teste, não para prever defeitos diretamente. **Correlaciona fortemente com LOC** (R² ≈ 0.93 em estudo empírico) — Shepperd (1988) mostra que em muitos casos é só um proxy para LOC. Usar essa correlação como exemplo central de violação de independência do Naive Bayes na Etapa 4. |
| 2 | Linhas de código (LOC) | Radon (`radon raw`) | Koru et al. (2008), *Theory of Relative Defect Proneness*: relação em lei de potência — módulos **menores** são proporcionalmente **mais** propensos a defeito (contraintuitivo). "Tamanho" no estudo é literalmente LOC. Nota metodológica: os autores evitaram calcular "densidade de defeitos" (bugs ÷ LOC) por gerar correlação negativa artificial; modelaram a probabilidade de defeito diretamente. Nosso rótulo é binário, então não corremos esse risco — mencionar como evidência de rigor na Etapa 4. Discretizar refletindo a relação não-linear. |
| 3 | Número de autores distintos | PyDriller (`git log`) | Matsumoto et al. (2010): métricas de desenvolvedor melhoram a predição. Ver também Weyuker, Ostrand & Bell (2008), "Do Too Many Cooks Spoil the Broth?". |
| 4 | Churn (frequência de alteração) | PyDriller | Nagappan & Ball (2005): usar churn **relativo** (proporcional ao tamanho do módulo), não absoluto — churn absoluto é fraco preditor e correlaciona com tamanho; a versão relativa foi desenhada para reduzir essa dependência. Ressalva: por ser uma razão, ainda existe risco teórico de correlação residual com tamanho (mesma lógica do artefato de densidade de defeitos) — mencionar como nuance, não como garantia. |
| 5 | Número de imports/dependências externas | Módulo `ast` da biblioteca padrão do Python (não é ferramenta externa) | Substitui DIT/acoplamento (removidas). Mede acoplamento leve sem depender de classe. Um estudo recente mostrou que **acoplamento (CBO) na verdade correlaciona fortemente com tamanho** — reforça que a remoção de DIT/CBO também foi positiva do ponto de vista estatístico, não só de ferramenta. |
| 6 | Cobertura de testes | Coverage.py | Evidência **contestada**: múltiplos estudos (incl. survey com 235 respostas de 7 organizações) encontram correlação nula ou muito fraca com defeitos. Não tratar como relação óbvia — mencionar a contestação na análise crítica. |

6 features atende ao mínimo exigido (6 a 8). Não adicionar mais por ora — tempo
é escasso e o conjunto já tem fundamentação e material crítico suficientes.

## Discretização das features (3 categorias cada)

| # | Feature | Baixo | Médio | Alto |
|---|---------|-------|-------|------|
| 1 | Complexidade ciclomática | ≤ 10 | 11–20 | > 20 |
| 2 | LOC | < 50 | 50–200 | > 200 |
| 3 | Nº de autores distintos | 1 | 2–3 | ≥ 4 |
| 4 | Churn relativo (% do arquivo alterado nos últimos 90 dias) | < 10% | 10–30% | > 30% |
| 5 | Nº de imports/dependências externas | 0–3 | 4–8 | > 8 |
| 6 | Cobertura de testes | < 50% | 50–80% | > 80% |

Atenção ao montar as probabilidades condicionais: não assumir "LOC Alto = mais
risco" isoladamente — pelo achado de Koru et al., a combinação LOC Baixo +
Complexidade Alta deve ter risco elevado (pequeno e denso = arriscado).

## Geração da massa de dados de treinamento (Etapa 2)
A atividade exige que os 100+ registros sejam **gerados por IA/sinteticamente**
— não coletados de repositórios reais. Abordagem: gerar via **script Python**
(não texto solto escrito por uma IA), programando deliberadamente os padrões
already levantados, para cumprir a exigência de dados "intencionais e
realistas" e alimentar a análise crítica da Etapa 4:

- Complexidade ciclomática e LOC nascem correlacionadas (Shepperd, 1988).
- Módulos pequenos com complexidade alta têm risco elevado (Koru et al., 2008
  — não assumir "mais LOC = mais risco" de forma linear).
- Churn tratado como taxa relativa ao tamanho, não valor absoluto (Nagappan &
  Ball, 2005).
- Nº de imports pode ter correlação leve com LOC, mas mantendo variação
  própria — não replicar a distribuição de LOC/complexidade.
- Distribuição de classes (sim/não) razoável — evitar desbalanceamento extremo.
- Ruído/variação controlada — nunca sorteio uniforme puro nem regra
  determinística perfeita.

Radon e PyDriller ficam reservados para uso **opcional e ilustrativo** — por
exemplo, rodar em um repositório pequeno do GitHub só para demonstrar no
relatório que a coleta real seria factível, sem substituir os dados sintéticos.

## Convenções de output
- Relatórios finais em `/relatorios/`, em **PDF** (via `reportlab` ou markdown +
  `pandoc`), mantendo também o `.md` fonte.
- Dados em `/dados/` (CSV).
- Código SQL do classificador em `/sql/`.
- Casos de teste e resultados em `/testes/`.
- Nomear arquivos com prefixo da etapa: `etapa1_...`, `etapa2_...`, etc.
- Cada etapa é entregável de forma independente — não adiantar conteúdo de
  etapas futuras dentro do relatório de uma etapa anterior.

## Estrutura de pastas
```
projeto/
├── CLAUDE.md
├── relatorios/
├── dados/
├── sql/
└── testes/
```

## Perguntas de defesa (referência rápida para a apresentação)

**Por que arquivo e não classe como módulo?**
Nenhuma das 6 features depende de conceito de classe (DIT/CBO foram
removidas); arquivo é unidade universal e evita ferramenta extra.

**Por que só 6 features e não mais?**
Atende ao mínimo da atividade; cada feature adicional testada (Halstead, CBO)
mostrou forte correlação com LOC, reduzindo diversidade real sem agregar
informação nova.

**Por que os dados são sintéticos e não reais?**
Exigência explícita da Etapa 2 do enunciado — pedir dados reais seria
descumprir a atividade, não uma limitação técnica.

**Qual a maior violação de independência do Naive Bayes no seu modelo?**
Complexidade ciclomática e LOC, com correlação empírica documentada de
R² ≈ 0.93 (praticamente a mesma informação estatisticamente).

**O modelo tem alguma feature fraca?**
Sim — cobertura de testes tem evidência contestada na literatura quanto à sua
relação com defeitos; declarado como limitação assumida, não ignorada.

## Observação final
Manter todo conteúdo gerado dentro do que o autor consegue explicar oralmente.
Preferir clareza e justificativa sobre sofisticação desnecessária.
