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

## DECISÃO DE PROJETO — "Naive Bayes puro" (features geradas independentes)

**Esta é a decisão mais importante deste documento e afeta todos os relatórios.**

Depois de considerar programar correlações entre features de propósito (para
demonstrar violações de independência empiricamente), a decisão final foi
**gerar todas as 6 features de forma estatisticamente independente entre si**
no dataset sintético. Ou seja: nenhuma feature depende do valor de outra
feature no gerador. Só o **rótulo** (SIM/NÃO) depende das features — isso é
esperado e correto, é literalmente o que o Naive Bayes modela (P(feature|classe)).

**Por que essa escolha:**
- Testa o classificador no cenário em que sua própria suposição central
  (independência condicional) é **verdadeira por construção** — o "Naive Bayes
  em sua forma pura", sem contaminar o experimento com violações plantadas.
- Simplifica o gerador, os relatórios e a defesa oral: menos nuances e
  ressalvas para justificar.
- É uma escolha de simulação legítima e deve ser **declarada como tal** — não
  afirmar que о mundo real é assim. A literatura (Shepperd, 1988; R² ≈ 0,93
  entre complexidade e LOC) documenta que, no mundo real, complexidade e LOC
  são correlacionadas. Nossos dados sintéticos optam por não reproduzir essa
  correlação, de propósito, para isolar o comportamento teórico do modelo.

**Consequência para a Seção "Violações de Independência" (Etapa 1) e para a
Reflexão Crítica (Etapa 4):** deixam de ser uma análise **empírica** (não há
violação nos nossos dados para medir) e passam a ser uma análise **teórica /
de limitação da simulação**. A tese correta agora é:

> "Testamos o Naive Bayes no cenário ideal em que sua suposição de
> independência é verdadeira. Nesse cenário, o modelo se comporta exatamente
> como a teoria prevê — o que é bom para validar a implementação, mas não
> reflete um cenário de implantação real, onde complexidade e LOC, por
> exemplo, seriam correlacionadas (Shepperd, 1988). Isso não é uma limitação
> encontrada nos dados, é uma limitação de realismo da simulação, assumida
> conscientemente para isolar e verificar o comportamento teórico do
> algoritmo."

**NÃO fazer:** não reintroduzir nenhuma dependência entre features no gerador
(nem tendência linear, nem subpopulação de interação, nem ruído correlacionado).
Todas as 6 features devem ser sorteios estatisticamente independentes uns dos
outros.

## As 6 features

| # | Feature | Ferramenta (uso real) | Fundamentação |
|---|---------|------------------------|----------------|
| 1 | Complexidade ciclomática | Radon (`radon cc`) | McCabe (1976): métrica de caminhos de execução, criada para apoiar desenho de testes. No mundo real correlaciona com LOC (Shepperd, 1988; R² ≈ 0,93) — nos nossos dados sintéticos, gerada independente por decisão de projeto (ver "Naive Bayes puro" acima). |
| 2 | Linhas de código (LOC) | Radon (`radon raw`) | Métrica clássica de tamanho de código. Discretizada em 3 faixas simples (não há mais efeito de interação plantado — ver decisão acima). |
| 3 | Número de autores distintos | PyDriller (`git log`) | Matsumoto et al. (2010): métricas de desenvolvedor melhoram a predição. Weyuker, Ostrand & Bell (2008), "Do Too Many Cooks Spoil the Broth?". |
| 4 | Churn (frequência de alteração) | PyDriller | Nagappan & Ball (2005): usar churn **relativo** ao tamanho do módulo, não absoluto. Gerado como taxa independente das demais features. |
| 5 | Número de imports/dependências externas | Módulo `ast` da biblioteca padrão do Python | Substitui DIT/acoplamento (removidas por dependerem de classe). Mede acoplamento leve. Gerado independente de LOC. |
| 6 | Cobertura de testes | Coverage.py | Evidência **contestada** na literatura: múltiplos estudos (incl. survey com 235 respostas de 7 organizações) encontram correlação nula ou muito fraca com defeitos (Inozemtseva & Holmes, 2014; Gren & Antinyan, 2017). Gerada independente de churn. |

6 features atende ao mínimo exigido (6 a 8). Não adicionar mais por ora.

## Discretização das features (3 categorias cada)

| # | Feature | Baixo | Médio | Alto |
|---|---------|-------|-------|------|
| 1 | Complexidade ciclomática | ≤ 10 | 11–20 | > 20 |
| 2 | LOC | < 50 | 50–200 | > 200 |
| 3 | Nº de autores distintos | 1 | 2–3 | ≥ 4 |
| 4 | Churn relativo (% do arquivo alterado nos últimos 90 dias) | < 10% | 10–30% | > 30% |
| 5 | Nº de imports/dependências externas | 0–3 | 4–8 | > 8 |
| 6 | Cobertura de testes | < 50% | 50–80% | > 80% |

Sem efeito de interação entre LOC e complexidade (ver decisão "Naive Bayes
puro"). Cada feature contribui para o rótulo de forma independente das demais.

## Geração da massa de dados de treinamento (Etapa 2)

A atividade exige que os 100+ registros sejam **gerados por IA/sinteticamente**
— não coletados de repositórios reais. Abordagem: gerar via **script Python**
(`dados/gerar_dados.py`), não texto solto escrito por uma IA.

**Modelo generativo (features 100% independentes entre si):**
- Cada uma das 6 features é sorteada de sua própria distribuição, **sem depender
  do valor de nenhuma outra feature**.
- O **rótulo** (SIM/NÃO) é a única coisa que depende das features: calcular um
  score de risco = soma de pesos (log-odds) por categoria de cada uma das 6
  features + ruído gaussiano; os k = round(N × 0,35) módulos de maior score
  recebem SIM.
- Sem subpopulação de interação, sem tendência cruzada entre features, sem
  ruído compartilhado entre colunas.
- Distribuição de classes razoável (evitar desbalanceamento extremo) —
  manter `PROPORCAO_DEFEITO ≈ 0,35`.
- Ruído/variação controlada em cada feature — nunca sorteio uniforme puro nem
  regra determinística perfeita.

**Validação esperada (Etapa 2):** a matriz de correlação de Pearson entre as 6
features numéricas deve sair com todos os valores de |r| próximos de 0 —
isso é o resultado **desejado e esperado**, não um erro. A interpretação da
matriz na Etapa 2 deve mudar de "confirma as correlações programadas" para
"confirma que as features foram geradas independentes, como planejado".

Radon e PyDriller ficam reservados para uso **opcional e ilustrativo** — por
exemplo, rodar em um repositório pequeno do GitHub só para demonstrar no
relatório que a coleta real seria factível, sem substituir os dados sintéticos.

## Casos de teste (Etapa 4) — atenção ao caso "d"

O caso **d** ("armadilha de Koru", LOC baixo + complexidade alta) perde a
premissa de interação plantada. Reformular seu propósito: em vez de "provar
que o modelo captura a interação pequeno-e-denso", o caso agora testa e
demonstra que **sem** dependência plantada entre as features, o modelo
simplesmente soma dois efeitos individuais moderados (LOC baixo tende a
log-odds próximo de neutro; complexidade alta tem log-odds positivo) — sem
nenhum "bônus" de interação. Isso é um resultado didático por si só: mostra
o Naive Bayes se comportando exatamente como a teoria prevê (soma de efeitos
marginais, sem interação), reforçando a nova tese da Reflexão Crítica.

## Convenções de output
- Relatórios finais em `/relatorios/`, em **PDF** (via HTML/CSS + Playwright,
  ou `reportlab`/`pandoc`), mantendo também o `.md` fonte.
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

**Por que as features foram geradas independentes, se a literatura diz que
complexidade e LOC se correlacionam no mundo real?**
Decisão deliberada de simulação: testar o Naive Bayes no cenário em que sua
própria suposição central é verdadeira, isolando o comportamento teórico do
algoritmo sem misturar com o efeito de uma violação plantada. Isso é
declarado explicitamente como limitação de realismo da simulação, não
escondido — ver Reflexão Crítica da Etapa 4.

**O modelo tem alguma feature fraca?**
Sim — cobertura de testes tem evidência contestada na literatura quanto à sua
relação com defeitos (Inozemtseva & Holmes, 2014; Gren & Antinyan, 2017);
declarado como limitação assumida, não ignorada.

**Se não há violação de independência nos dados, qual é a "crítica" da Etapa 4?**
A crítica muda de "o modelo erra porque os dados violam a suposição" para
"testamos o modelo no cenário ideal onde a suposição é verdadeira; ele se
comporta como a teoria prevê, mas isso não reflete um cenário real de
implantação, onde essa independência não existiria de fato". A limitação
discutida é sobre o **realismo da simulação**, não sobre um erro do modelo.

## Observação final
Manter todo conteúdo gerado dentro do que o autor consegue explicar oralmente.
Preferir clareza e justificativa sobre sofisticação desnecessária.
