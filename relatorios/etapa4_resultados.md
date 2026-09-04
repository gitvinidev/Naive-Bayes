# Atividade Prática 1 — Classificador Bayesiano
## Etapa 4 — Resultados dos Testes e Reflexão Crítica

| | |
|---|---|
| **Autor** | Marcus Viníicius Santos de Almeida |
| **Disciplina** | Mineração de Dados |
| **Data de entrega** | 04/09/2026 |
| **Cobre os entregáveis** | "Resultados dos testes" (5+ casos com análise) e "Reflexão crítica" |
| **Artefatos** | `testes/casos_teste.csv` · `testes/rodar_casos_teste.py` · `testes/resultados_casos.txt` |
| **Modelo de treino** | 150 módulos sintéticos da Etapa 2 (decisão "Naive Bayes puro" — features independentes entre si) · P(SIM) = 52/150 = 0,347 · P(NAO) = 98/150 = 0,653 |

Todos os números deste relatório saem de `python3 testes/rodar_casos_teste.py`
(que reconstrói o banco, roda os 6 casos e calcula os log-odds a partir da view
`verossimilhancas` da Etapa 3), rodado sobre a massa de dados da versão atual
da Etapa 2 — 6 features geradas independentes entre si (ver `CLAUDE.md`, seção
"Naive Bayes puro", e o relatório da Etapa 2, §1.2).

---

## 1. Os 6 casos de teste

Cada caso usa as 6 features e as faixas `baixo`/`medio`/`alto` do CLAUDE.md. Os
perfis são os mesmos da versão anterior deste relatório — o que mudou foi
apenas a massa de treino sobre a qual eles são classificados.

| Caso | Propósito | complexidade | loc | n_autores | churn | n_imports | cobertura | Intuição |
|---|---|---|---|---|---|---|---|---|
| **a** | Baixo risco claro — tudo bom | baixo | medio | baixo | baixo | baixo | alto | BAIXO |
| **b** | Alto risco claro — tudo ruim | alto | alto | alto | alto | alto | baixo | ALTO |
| **c** | Ambíguo — sinais contraditórios (complexidade alta **e** cobertura alta) | alto | medio | medio | medio | medio | alto | ~50/50 |
| **d** | Armadilha de Koru, sem bônus de interação — arquivo pequeno **e** denso | alto | baixo | medio | medio | medio | medio | ALTO |
| **e** | Combinação rara / fronteira — arquivo grande porém simples, churn alto | baixo | alto | baixo | alto | baixo | alto | BAIXO/indefinido |
| **f** | Feature contestada — tudo ruim, **mas cobertura alta** | alto | alto | alto | alto | alto | alto | ALTO |

---

## 2. Resultados

| Caso | P(SIM) | P(NAO) | Recomendação | Bate com a intuição? |
|---|---|---|---|---|
| **a** | 4,28% | 95,72% | BAIXO RISCO | **Sim** |
| **b** | 95,53% | 4,47% | ALTO RISCO | **Sim** |
| **c** | 38,04% | 61,96% | BAIXO RISCO | **Parcial** — perto de 50/50 como esperado, mas agora do lado NÃO (era ALTO na versão anterior deste relatório; ver discussão abaixo) |
| **d** | 55,52% | 44,48% | ALTO RISCO | **Sim, mas por pouco** — sem bônus de interação, o resultado fica bem mais perto da fronteira do que na versão anterior (era 74,40%) |
| **e** | 22,14% | 77,86% | BAIXO RISCO | **Sim** (com ressalva sobre confiança — ver §4.3) |
| **f** | 92,64% | 7,36% | ALTO RISCO | **Sim** — cobertura alta reduz um pouco o risco frente ao caso b, mas não o suficiente para "salvar" o módulo |

As duas probabilidades somam 100,00 % em todos os casos e as 6 features casaram
com a tabela de verossimilhanças (`n_features = 6/6`).

### Como cada resultado se forma

O classificador é, na prática, um **somatório de log-odds**: começa no log-odds
a priori `ln(52/98) = −0,633` (a base já pende para NAO) e soma o log-odds de
cada categoria observada (tabela do §3). `P(SIM) > 50 %` ⇔ soma total > 0.

- **Caso a** — soma das features = −2,47; com o prior, total −3,11 → P(SIM) 4 %.
  Todas as categorias "boas" têm log-odds negativo; o modelo acumula evidência
  contra defeito. Correto.
- **Caso b** — soma das features = +3,70; total +3,06 → P(SIM) 96 %. Espelho do
  caso a. Correto.
- **Caso c** — soma das features = +0,15; total −0,49 → P(SIM) 38 %. A
  `complexidade = alto` sozinha vale **+0,76**, mas `n_autores = medio` (−0,14),
  `n_imports = medio` (−0,07) e sobretudo `loc = medio` (−0,26) e
  `cobertura = alto` (−0,15) puxam na direção contrária o suficiente para virar
  o placar. Na versão anterior deste relatório (dados com correlações
  plantadas) esse mesmo perfil dava ALTO RISCO (61 %); aqui dá BAIXO RISCO
  (38 %). Não é um erro: é o resultado esperado de mudar os dados de treino —
  os pesos por categoria de cada feature mudaram porque a massa de dados é
  outra, e o caso continua sendo, como o nome propõe, o mais perto de 50/50
  dos seis.
- **Caso d** — soma das features = +0,86; total +0,22 → P(SIM) 55,5 %. Aqui está
  o ponto de Koru, sem bônus de interação: `loc = baixo` tem log-odds **+0,45**
  (empurra para SIM, não para NAO) e `complexidade = alto` tem **+0,76** —
  juntas somam +1,21, e as outras 4 features (todas `medio`) somam −0,35,
  resultando no total acima. O modelo classifica o arquivo pequeno-e-denso
  como ALTO RISCO, mas agora **por pouco** (55,5 % vs. 44,5 %) — não pelos
  74,4 % da versão anterior, que incluía um bônus de interação plantado de
  propósito no gerador antigo. Discussão completa em §4.1.
- **Caso e** — soma das features = −0,62; total −1,26 → P(SIM) 22 %. O
  `churn = alto` (+1,20) puxa para SIM, mas é superado por `complexidade =
  baixo` (−0,61), `n_autores = baixo` (−0,93), `n_imports = baixo` (−0,27) e
  `cobertura = alto` (−0,15). Resultado: BAIXO RISCO. Discussão em §4.3.
- **Caso f** — soma das features = +3,17; total +2,53 → P(SIM) 93 %. É o caso b
  com `cobertura` trocada de `baixo` para `alto`. O resultado **caiu** de
  95,53 % para 92,64 %: no treino atual, `cobertura = baixo` tem log-odds
  +0,37 e `cobertura = alto` tem −0,15 — dessa vez cobertura se comporta na
  direção "intuitiva" (mais cobertura, menos risco), ao contrário da versão
  anterior deste relatório, em que o efeito ia na direção contrária. Mesmo
  assim, o módulo continua ALTO RISCO com folga: a diferença entre os dois
  casos (≈0,52 de log-odds) é pequena perto da soma das outras 5 features
  ruins (≈+3,17). Ver §4.4 e §5.

---

## 3. Poder discriminativo das features (log-odds)

**Log-odds de uma categoria** = `ln( P(categoria | SIM) / P(categoria | NAO) )`,
o "peso de evidência" daquela categoria: positivo empurra a classificação para
SIM, negativo para NAO, e |valor| grande = categoria informativa.

### Log-odds por categoria

| Feature | `baixo` | `medio` | `alto` | Formato |
|---|---|---|---|---|
| **n_autores** | −0,93 | −0,14 | **+1,01** | monótono forte |
| **complexidade** | −0,61 | +0,41 | **+0,76** | monótono |
| **churn** | −0,25 | +0,01 | **+1,20** | monótono, salto grande no `alto` |
| **loc** | +0,45 | −0,26 | +0,15 | **em U** (baixo *e* alto arriscados, mediano protege) |
| **cobertura** | +0,37 | −0,15 | **−0,15** | monótono fraco (direção "intuitiva" desta vez) |
| **n_imports** | −0,27 | −0,07 | +0,20 | monótono fraco |

### Ranking por |log-odds| médio (poder discriminativo)

| # | Feature | \|log-odds\| médio | \|log-odds\| máx |
|---|---|---|---|
| 1 | **nº de autores** | 0,695 | 1,013 |
| 2 | complexidade ciclomática | 0,595 | 0,762 |
| 3 | churn relativo | 0,482 | 1,196 |
| 4 | LOC | 0,285 | 0,445 |
| 5 | cobertura de testes | 0,227 | 0,374 |
| 6 | nº de imports | 0,180 | 0,268 |

Note-se que o ranking **mudou de ordem** em relação à versão anterior deste
relatório (lá: complexidade > imports > autores > loc > churn > cobertura).
Isso é esperado: os pesos de rótulo por categoria (`PESOS` no gerador) são os
mesmos de antes, mas agora cada feature é sorteada de forma independente das
demais, então a *frequência exata* de cada categoria dentro de cada classe —
e portanto o log-odds calculado sobre a amostra de 150 módulos — depende só do
sorteio daquela feature, não mais de covariar com outra. Cobertura de testes
continua entre as duas features menos discriminativas, consistente com a
literatura contestada sobre essa métrica.

---

## 4. Análise — as 4 perguntas da atividade

### 4.1 (a) O modelo classificou conforme a intuição do domínio?

**Em geral, sim, com um contraste pedagógico importante no caso d.** Os dois
casos "extremos" (a e b) saíram exatamente como o domínio prevê. Caso a caso:

- **Casos a e b** (baixo/alto risco claros): acertos limpos, 4 % e 96 % de
  P(SIM). O modelo empilha evidência coerente.
- **Caso c** (ambíguo): o resultado (38 %, BAIXO RISCO) inverteu de lado frente
  à versão anterior deste relatório (61 %, ALTO RISCO), mas continua sendo,
  dos seis, o mais perto de 50/50 — a marca de um caso genuinamente ambíguo
  não é sempre cair do mesmo lado, é cair perto da fronteira. A mudança de
  lado é consequência de trocar a massa de treino (Etapa 2), não um defeito do
  classificador.
- **Caso d — a "armadilha de Koru", agora sem bônus de interação plantado.**
  Este é o caso mais importante para entender a diferença entre esta versão do
  projeto e a anterior. Na versão anterior, o gerador da Etapa 2 somava um
  **bônus explícito** de +0,70 ao score de risco sempre que `loc = baixo` **e**
  `complexidade = alto` ocorriam juntas — um efeito de interação plantado de
  propósito, e o resultado era P(SIM) = 74,40 %. Nesta versão, essa dependência
  entre features foi removida do gerador (decisão "Naive Bayes puro" — Etapa 2,
  §1.2): o rótulo continua dependendo de cada feature, mas **não há mais
  nenhum termo que dependa da combinação das duas**. O resultado (P(SIM) =
  55,52 %) é, por construção, exatamente a soma dos dois efeitos individuais —
  o log-odds de `loc = baixo` isolado (+0,45) mais o log-odds de
  `complexidade = alto` isolado (+0,76), ajustado pelas outras 4 features e
  pelo prior — **sem nenhum bônus extra**. O modelo ainda classifica o arquivo
  pequeno-e-denso como ALTO RISCO (a soma dos dois efeitos positivos é
  suficiente para passar de 50 %), mas por uma margem pequena, bem diferente
  da confiança de 74 % da versão anterior. Isso demonstra exatamente o que o
  Naive Bayes faz e não faz: ele **soma evidências marginais**, nunca modela um
  efeito de interação que exceda a soma das partes — quando não há interação
  plantada nos dados, o resultado reflete apenas essa soma, sem "descontos" nem
  "bônus" adicionais.
- **Caso e** (raro): resultado plausível (BAIXO RISCO, 22 %), com a ressalva de
  §4.3 sobre o modelo soar confiante para um perfil nunca visto no treino.
- **Caso f** (cobertura contestada): a cobertura alta reduziu discretamente o
  risco frente ao caso b (95,53 % → 92,64 %), mas não o suficiente para tirar o
  módulo de ALTO RISCO — as outras 5 features ruins dominam. Ao contrário da
  versão anterior deste relatório, aqui a cobertura se comporta na direção
  "intuitiva" (mais cobertura, menos risco); mesmo assim seu efeito é pequeno
  frente às demais features, consistente com a literatura que trata essa
  relação como fraca/contestada (§4.2).

### 4.2 (b) Quais features tiveram maior log-odds (maior poder discriminativo)?

Ranking do §3:

1. **Nº de autores distintos** lidera desta vez (|log-odds| médio 0,695; a
   categoria `alto` sozinha vale +1,01).
2. **Complexidade ciclomática** vem em seguida (0,595; `alto` = +0,76).
3. **Churn relativo** tem o maior valor máximo isolado da tabela (`alto` =
   +1,20), mas média um pouco menor (0,482) porque `baixo` e `medio` pesam
   pouco.
4. **LOC** é o caso mais interessante estruturalmente: o log-odds é **em U** —
   `baixo` (+0,45) e `alto` (+0,15) empurram para SIM, `medio` (−0,26) para
   NAO. O modelo continua capturando a não-linearidade de Koru (§4.1, caso d),
   e isso não depende de LOC estar correlacionado com nenhuma outra feature —
   é uma propriedade da relação entre LOC e o rótulo, isolada.
5. **Cobertura de testes** é a **segunda menos discriminativa** (|log-odds|
   médio 0,227), com efeito pequeno e agora na direção "intuitiva" — ainda
   assim, pequeno demais para mudar uma classificação por conta própria (caso
   f). Consistente com a literatura que trata a relação cobertura↔defeitos
   como contestada (Inozemtseva & Holmes, 2014; Gren & Antinyan, 2017).
6. **Nº de imports** é a **menos discriminativa** de todas (|log-odds| médio
   0,180; máximo 0,268).

O fato de o ranking ter mudado de ordem frente à versão anterior (lá,
complexidade liderava com folga) é esperado e reforça a Seção 5: quando as
features passam a ser sorteadas de forma independente, cada uma carrega
exatamente a informação da sua própria distribuição — nenhuma "empresta" poder
discriminativo de outra por estarem correlacionadas.

### 4.3 (c) O caso de valor não visto / na fronteira (caso e) e o papel de Laplace

O perfil do caso e (`complexidade baixo, loc alto, n_autores baixo, churn alto,
n_imports baixo, cobertura alto`) **não aparece em nenhum dos 150 módulos de
treino** (`perfil exato no treino = 0`). Ainda assim o classificador devolve uma
resposta bem definida: P(NAO) = 77,9 %.

**Por que ele consegue responder.** O Naive Bayes nunca precisa ter visto o
perfil completo — pela hipótese de independência, ele só usa as **seis fatias
unidimensionais** `P(categoria | classe)`, e cada uma dessas fatias está bem
povoada. Ou seja, a esparsidade da combinação de 6 categorias (729 perfis
possíveis, 150 exemplos) não trava o modelo. Essa é a **força** do Naive Bayes —
e também o seu risco: ele soa confiante (78/22) sobre uma combinação que talvez
mereça mais incerteza justamente por nunca ter sido observada por inteiro.

**O que a suavização de Laplace faz aqui.** Com este dataset, a categoria mais
rara de todas é `n_autores = baixo` na classe SIM: apenas 6 dos 52 módulos com
defeito. Sem suavização, `P(n_autores=baixo|SIM)` seria `6/52 ≈ 0,115`; com
Laplace, `(6+1)/(52+3) = 7/55 ≈ 0,127`. Nenhuma célula `(feature, categoria,
classe)` deste dataset está vazia, então Laplace não precisou impedir nenhum
`ln(0)` neste teste específico — o que ele fez foi encolher levemente cada
estimativa na direção do uniforme, como sempre faz.

**Quando Laplace seria decisivo.** Se tivéssemos discretizado em mais faixas,
ou com menos dados, alguma célula ficaria em 0 e — sem suavização — `ln(0) =
−∞` zeraria a classe inteira por causa de **uma** categoria não observada. Com
Laplace essa célula valeria `1/55 ≈ 0,018` em vez de 0, e a classe continuaria
"viva" para as outras 5 features decidirem. É uma apólice de seguro que, neste
conjunto de dados específico, não precisou ser acionada — mas que mantém o
modelo bem-definido para qualquer entrada.

### 4.4 (d) Limitações do Naive Bayes neste domínio

1. **A independência entre features testada aqui é verdadeira por construção,
   não por realismo.** Esta é a limitação central desta versão do projeto, e é
   diferente da limitação discutida na versão anterior. Os dados de treino da
   Etapa 2 foram gerados com as 6 features estatisticamente independentes
   entre si (decisão "Naive Bayes puro" — CLAUDE.md), e a matriz de correlação
   da Etapa 2 confirma isso (todos os |r| < 0,10). Isso significa que **não
   há**, neste dataset, nenhuma violação empírica de independência para medir
   — e a análise abaixo não afirma ter encontrado uma. O que existe é uma
   limitação **teórica, documentada na literatura**: em código real,
   complexidade ciclomática e LOC são fortemente correlacionadas (R² ≈ 0,93 —
   Shepperd, 1988; Seção 5 do relatório da Etapa 1). Um classificador treinado
   sobre dados sintéticos independentes se comporta exatamente como a teoria
   do Naive Bayes prevê (soma de evidências marginais, sem "contar duas
   vezes"), mas isso não garante o mesmo comportamento sobre uma massa de
   dados real, onde essa correlação existiria de fato. A distinção importa:
   é uma limitação **do realismo da simulação**, não um erro encontrado nos
   dados nem um erro de implementação do classificador.
2. **O Naive Bayes não modela interação entre features — nem para mais, nem
   para menos.** O caso d (§4.1) evidencia isso de forma limpa: sem nenhum
   termo de interação nos dados, o resultado para "pequeno e denso" é
   exatamente a soma dos dois efeitos individuais (LOC baixo + complexidade
   alta), nem mais nem menos. Se, em um cenário real, o efeito combinado de
   "pequeno e denso" fosse **maior** do que a soma das partes (o que a teoria
   de Koru et al. sugere ser plausível), o Naive Bayes subestimaria esse
   risco — não porque os dados de treino "escondam" a interação, mas porque o
   modelo, por definição, não tem como representar termos de interação.
3. **Relação não-linear de LOC com risco.** Isso continua valendo e **não**
   depende de LOC estar correlacionado com outra feature: a discretização em 3
   faixas captura a forma em U (§3) porque o peso atribuído a `loc = baixo` na
   geração dos dados é individualmente maior que o de `loc = medio` — uma
   propriedade de LOC isolado, coerente com Koru et al. (2008).
4. **Feature de evidência contestada: cobertura de testes.** Continua entre as
   duas features menos discriminativas do ranking (§3), com efeito pequeno
   sobre o score final (caso f). Mantê-la é uma limitação assumida e
   documentada, não um descuido.
5. **Simplificação módulo = arquivo (e não classe).** Inalterada frente à
   versão anterior: o estudo de referência (Koru et al.) usa classe como
   unidade; adotamos arquivo por ser universal e não exigir parser por
   linguagem, ao custo de métricas mais "grossas" por registro.
6. **Dados de treino sintéticos.** Os 150 módulos foram gerados por script
   (Etapa 2). Diferente da versão anterior, o rótulo aqui depende das 6
   features de forma independente, mas o rótulo em si ainda é definido por
   pesos escolhidos pelo autor a partir da literatura — a validação contra a
   "intuição do domínio" continua, em parte, circular quanto a essa direção
   individual de cada feature (mesmo não sendo mais circular quanto à
   interação entre elas).

---

## 5. Reflexão Crítica

Este projeto testou o classificador Naive Bayes exatamente no cenário em que a
sua própria suposição central — independência condicional entre features dado
o rótulo — é **verdadeira por construção dos dados**. Os resultados confirmam
que, nesse cenário, o modelo se comporta **exatamente como a teoria prevê**:
separa com folga os módulos claramente arriscados dos claramente seguros
(casos a e b, 4 % vs 96 %), responde de forma bem definida a um perfil nunca
visto no treino porque só depende de fatias unidimensionais bem povoadas (caso
e, §4.3), e, no caso mais revelador (caso d), produz exatamente a **soma** dos
dois efeitos marginais de "LOC baixo" e "complexidade alta" — nem mais, nem
menos — porque não há, nos dados, nenhum termo de interação para o modelo
capturar ou deixar de capturar.

Isso é bom e é ruim ao mesmo tempo, e é importante não confundir as duas
coisas. É **bom** porque valida a implementação: o classificador SQL soma
priors e verossimilhanças em log corretamente, aplica Laplace corretamente e
produz decisões coerentes com os pesos que geraram os dados — se houvesse um
bug na Etapa 3, este seria o cenário mais fácil de expor um comportamento
errado, e não expôs nenhum. É **limitado** porque este é o cenário mais
favorável possível para um Naive Bayes, e não é o cenário de um sistema real
em produção. Em código real, complexidade ciclomática e LOC são correlacionadas
com força (R² ≈ 0,93 — Shepperd, 1988), e o Naive Bayes, ao tratá-las como
independentes, contaria a evidência de "tamanho/estrutura do código" duas
vezes, ficando mais confiante do que deveria exatamente nos casos onde essas
features concordam. Este projeto **não mediu** esse efeito porque optou, de
propósito, por não reproduzi-lo nos dados sintéticos (Etapa 2, §1.2) — então é
preciso ser preciso sobre o que está sendo afirmado aqui: isto é uma limitação
**teórica, documentada na literatura consultada na Etapa 1**, não uma
violação **encontrada empiricamente** nesta massa de dados. A matriz de
correlação da Etapa 2 saiu com todos os |r| abaixo de 0,10 — não há, neste
dataset, nada para "descobrir" nesse sentido.

O veredito honesto: o modelo implementado está correto e bem testado dentro do
cenário para o qual foi desenhado — um Naive Bayes "em sua forma pura". Antes
de usá-lo sobre dados reais de um repositório de verdade, seria necessário
reavaliar a suposição de independência sobre a massa real (medir a correlação
entre complexidade e LOC nesse repositório, por exemplo) e, se ela se
confirmar como na literatura, considerar reduzir a redundância (fundir as duas
features, ou trocar por um modelo que aceite correlação, como regressão
logística) antes de confiar nas probabilidades absolutas — não só na ordenação
relativa dos módulos — que o classificador produz.

---

### Nota sobre o uso de IA

Os casos de teste, o script de execução e esta análise foram construídos em
diálogo com uma IA generativa. Cada resultado numérico foi reproduzido pelo
script (`testes/rodar_casos_teste.py`) e conferido contra a decomposição em
log-odds; as limitações teóricas foram checadas contra as referências
levantadas na Etapa 1. A tese central desta versão do relatório — a distinção
entre "limitação teórica documentada na literatura" e "violação encontrada
empiricamente nos dados" — foi definida como decisão de projeto ("Naive Bayes
puro", `CLAUDE.md`) antes da geração dos dados, não inferida depois de olhar os
resultados. Todo o conteúdo é defensável oralmente.
