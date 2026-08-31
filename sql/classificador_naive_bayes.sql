-- ======================================================================
--  CLASSIFICADOR NAIVE BAYES EM SQL  —  Etapa 3
--  Domínio: predição de defeitos de software a partir de métricas estáticas.
--  Banco:   SQLite (dados/classificador.db)
-- ======================================================================
--
--  Este script define, NESTA ORDEM:
--    (a) priors             — P(classe)
--    (b) treino_longo        — dados de treino em formato longo (unpivot)
--    (c) verossimilhancas    — P(categoria | classe) com suavização de Laplace
--    (d) score_log           — soma de log-probabilidades por classe
--    (e) classificar_modulo  — normaliza os log-scores em probabilidade 0–100%
--                              e emite uma recomendação textual
--
--  Pré-requisito: a tabela `dados_treinamento` já deve existir e estar
--  carregada com o CSV da Etapa 2 (feito por sql/rodar_classificador.py).
--
--  Funções LN e EXP: nativas no SQLite >= 3.35 compilado com
--  SQLITE_ENABLE_MATH_FUNCTIONS. O ambiente-alvo (SQLite 3.45.1 do módulo
--  sqlite3 do Python) já as traz. Como garantia de portabilidade,
--  sql/rodar_classificador.py testa `SELECT ln(1)` e, se falhar, registra
--  `ln` e `exp` como funções Python (math.log / math.exp) antes de rodar
--  este script — ver comentário lá.
--
--  Todo o script é idempotente: cada objeto é derrubado (DROP ... IF EXISTS)
--  antes de ser recriado, então pode ser reexecutado à vontade.
-- ======================================================================


-- ----------------------------------------------------------------------
-- (a) VIEW priors  —  probabilidades a priori P(classe)
-- ----------------------------------------------------------------------
-- Contagem simples de cada rótulo em dados_treinamento, dividida pelo
-- total de módulos. É o "chute inicial" antes de olhar as features:
-- se 35% dos módulos históricos têm defeito, P(SIM) = 0,35.
--
-- Colunas expostas:
--   classe    — 'SIM' ou 'NAO'
--   n_classe  — nº de módulos de treino dessa classe
--   n_total   — nº total de módulos de treino
--   p_prior   — n_classe / n_total  (probabilidade a priori)
DROP VIEW IF EXISTS priors;
CREATE VIEW priors AS
SELECT
    defeito                                             AS classe,
    COUNT(*)                                            AS n_classe,
    (SELECT COUNT(*) FROM dados_treinamento)            AS n_total,
    CAST(COUNT(*) AS REAL)
        / (SELECT COUNT(*) FROM dados_treinamento)      AS p_prior
FROM dados_treinamento
GROUP BY defeito;


-- ----------------------------------------------------------------------
-- (b) VIEW treino_longo  —  unpivot das 6 colunas de categoria
-- ----------------------------------------------------------------------
-- dados_treinamento é "largo": 1 linha por módulo, com 6 colunas de
-- categoria (complexidade_cat, loc_cat, ...). Para calcular a
-- verossimilhança de cada feature sem escrever a mesma lógica 6 vezes,
-- transformamos para o formato "longo":
--
--     (modulo_id, defeito, feature, categoria)   -- 1 linha por módulo x feature
--
-- 150 módulos x 6 features = 900 linhas. O UNION ALL empilha os 6
-- "recortes" (um por feature). Os nomes curtos de feature abaixo
-- ('complexidade', 'loc', ...) são a chave usada no resto do script e
-- também nos casos de teste (tabela caso_teste).
DROP VIEW IF EXISTS treino_longo;
CREATE VIEW treino_longo AS
    SELECT modulo_id, defeito, 'complexidade' AS feature, complexidade_cat      AS categoria FROM dados_treinamento
    UNION ALL
    SELECT modulo_id, defeito, 'loc',                     loc_cat                           FROM dados_treinamento
    UNION ALL
    SELECT modulo_id, defeito, 'n_autores',               n_autores_cat                     FROM dados_treinamento
    UNION ALL
    SELECT modulo_id, defeito, 'churn',                   churn_relativo_cat                FROM dados_treinamento
    UNION ALL
    SELECT modulo_id, defeito, 'n_imports',               n_imports_cat                     FROM dados_treinamento
    UNION ALL
    SELECT modulo_id, defeito, 'cobertura',               cobertura_testes_cat              FROM dados_treinamento;


-- ----------------------------------------------------------------------
-- (c) VIEW verossimilhancas  —  P(categoria | classe) com Laplace
-- ----------------------------------------------------------------------
-- Para cada combinação (feature, categoria, classe) queremos:
--
--     P(feature = categoria | classe)
--         = (contagem observada + 1) / (total da classe + k)
--
-- onde k = 3 = nº de categorias possíveis (baixo / medio / alto),
-- igual para as 6 features.
--
-- POR QUE somar 1 no numerador e k no denominador (suavização de Laplace):
--   Sem isso, se uma categoria nunca aparece junto com uma classe no
--   treino, a contagem é 0 e P(categoria|classe) = 0. Como o Naive Bayes
--   MULTIPLICA as probabilidades (aqui: soma os logs), um único zero
--   zera o score inteiro daquela classe — o modelo fica "certo demais"
--   por causa de uma célula vazia que é só falta de dados. Laplace
--   distribui uma "meia-observação" para cada categoria: nenhuma
--   probabilidade é exatamente 0 nem exatamente 1, e o efeito sobre as
--   categorias bem povoadas é desprezível.
--
-- A CTE `grade` gera TODAS as 6*3*2 = 36 combinações possíveis via
-- CROSS JOIN. O LEFT JOIN com as contagens observadas devolve 0
-- (COALESCE) para as combinações que não aparecem no treino — é aí que
-- o Laplace age.
DROP VIEW IF EXISTS verossimilhancas;
CREATE VIEW verossimilhancas AS
WITH
    classes(classe) AS (
        SELECT 'SIM' UNION ALL SELECT 'NAO'
    ),
    features(feature) AS (
        SELECT 'complexidade' UNION ALL SELECT 'loc'       UNION ALL SELECT 'n_autores'
        UNION ALL SELECT 'churn' UNION ALL SELECT 'n_imports' UNION ALL SELECT 'cobertura'
    ),
    categorias(categoria) AS (
        SELECT 'baixo' UNION ALL SELECT 'medio' UNION ALL SELECT 'alto'
    ),
    -- todas as combinações possíveis (mesmo as que não ocorrem no treino)
    grade AS (
        SELECT f.feature, cat.categoria, c.classe
        FROM features f
        CROSS JOIN categorias cat
        CROSS JOIN classes c
    ),
    -- contagem observada de cada (feature, categoria, classe) no treino longo
    contagens AS (
        SELECT feature, categoria, defeito AS classe, COUNT(*) AS n
        FROM treino_longo
        GROUP BY feature, categoria, defeito
    )
SELECT
    g.feature,
    g.categoria,
    g.classe,
    COALESCE(ct.n, 0)                                              AS contagem,
    p.n_classe                                                     AS total_classe,
    3                                                             AS k_laplace,
    -- fórmula de Laplace:  (contagem + 1) / (total_da_classe + k)
    CAST(COALESCE(ct.n, 0) + 1 AS REAL) / (p.n_classe + 3)         AS p_verossimilhanca
FROM grade g
LEFT JOIN contagens ct
       ON ct.feature   = g.feature
      AND ct.categoria = g.categoria
      AND ct.classe    = g.classe
JOIN priors p
       ON p.classe = g.classe;


-- ----------------------------------------------------------------------
--  Tabela de ENTRADA: caso_teste  —  o(s) módulo(s) a classificar
-- ----------------------------------------------------------------------
-- Um módulo novo entra como 6 linhas no formato longo:
--     (caso_id, feature, categoria)
-- 'caso_id' identifica o caso (permite classificar vários de uma vez).
-- 'feature' usa as mesmas chaves de treino_longo/verossimilhancas.
-- sql/rodar_classificador.py preenche esta tabela.
CREATE TABLE IF NOT EXISTS caso_teste (
    caso_id   TEXT NOT NULL,
    feature   TEXT NOT NULL,
    categoria TEXT NOT NULL
);


-- ----------------------------------------------------------------------
-- (d) VIEW score_log  —  log-score de cada classe para cada caso
-- ----------------------------------------------------------------------
-- Naive Bayes: para cada classe,
--
--     score(classe) = P(classe) * PRODUTO_i P(feature_i = cat_i | classe)
--
-- POR QUE usar logaritmo em vez de multiplicar direto:
--   Cada P(...) é um número < 1 (aqui tipicamente entre 0,02 e 0,7).
--   Multiplicar 7 números pequenos (1 prior + 6 features) gera valores
--   minúsculos; com muitas features isso chega a "underflow" (o float
--   vira 0 e perde-se a informação). Tomando log, o PRODUTO vira SOMA:
--
--     log score(classe) = ln P(classe) + SOMA_i ln P(cat_i | classe)
--
--   Somar números da ordem de -1 a -4 é numericamente seguro, e como
--   log é monotônico a classe de maior log-score é a mesma de maior
--   probabilidade. A conversão de volta para % é feita na view (e).
--
-- O JOIN liga cada uma das 6 linhas do caso às verossimilhanças das DUAS
-- classes, então o GROUP BY produz 2 linhas por caso (SIM e NAO).
--   n_features   — deve ser 6; se vier menor, alguma categoria do caso
--                  não casou (nome de feature errado ou categoria fora
--                  de baixo/medio/alto).
--   log_prior    — ln P(classe); é constante dentro do grupo, então
--                  MIN() apenas "extrai" esse valor único para dentro do
--                  agregado (poderia ser MAX/AVG — dá o mesmo).
DROP VIEW IF EXISTS score_log;
CREATE VIEW score_log AS
SELECT
    ct.caso_id,
    v.classe,
    COUNT(*)                                              AS n_features,
    MIN(ln(p.p_prior))                                    AS log_prior,
    SUM(ln(v.p_verossimilhanca))                          AS soma_log_veros,
    MIN(ln(p.p_prior)) + SUM(ln(v.p_verossimilhanca))     AS log_score
FROM caso_teste ct
JOIN verossimilhancas v
       ON v.feature   = ct.feature
      AND v.categoria = ct.categoria
JOIN priors p
       ON p.classe = v.classe
GROUP BY ct.caso_id, v.classe;


-- ----------------------------------------------------------------------
-- (e) VIEW classificar_modulo  —  normalização + recomendação
-- ----------------------------------------------------------------------
-- Converte os dois log-scores (SIM e NAO) de volta para probabilidade
-- entre 0 e 100%:
--
--     P(SIM) = exp(log_sim) / ( exp(log_sim) + exp(log_nao) )
--
-- (o denominador P(caso) some porque aparece igual nas duas classes;
--  reexponenciar e dividir pela soma reintroduz a normalização).
--
-- Estabilidade numérica: subtraímos log_max = max(log_sim, log_nao) dos
-- dois expoentes antes do exp(). Isso NÃO muda o resultado (o fator
-- exp(-log_max) cancela no numerador e no denominador), só evita que
-- exp() receba argumentos muito negativos. Para 7 fatores como aqui
-- seria dispensável; deixamos explícito porque é a forma correta de
-- fazer a conta e é fácil de justificar.
--
-- Saída (1 linha por caso):
--   prob_sim_pct, prob_nao_pct  — somam 100
--   recomendacao                — texto de decisão baseado em P(SIM) > 50%
DROP VIEW IF EXISTS classificar_modulo;
CREATE VIEW classificar_modulo AS
WITH
    -- log-scores das duas classes lado a lado, uma linha por caso
    pivo AS (
        SELECT
            caso_id,
            MAX(n_features)                                        AS n_features,
            MAX(CASE WHEN classe = 'SIM' THEN log_score END)       AS log_sim,
            MAX(CASE WHEN classe = 'NAO' THEN log_score END)       AS log_nao
        FROM score_log
        GROUP BY caso_id
    ),
    est AS (
        SELECT
            caso_id, n_features, log_sim, log_nao,
            MAX(log_sim, log_nao)                                  AS log_max
        FROM pivo
    ),
    exps AS (
        SELECT
            caso_id, n_features, log_sim, log_nao,
            exp(log_sim - log_max)                                 AS e_sim,
            exp(log_nao - log_max)                                 AS e_nao
        FROM est
    )
SELECT
    caso_id,
    n_features,
    ROUND(log_sim, 4)                                              AS log_score_sim,
    ROUND(log_nao, 4)                                              AS log_score_nao,
    ROUND(100.0 * e_sim / (e_sim + e_nao), 2)                      AS prob_sim_pct,
    ROUND(100.0 * e_nao / (e_sim + e_nao), 2)                      AS prob_nao_pct,
    CASE
        WHEN e_sim / (e_sim + e_nao) > 0.5
            THEN 'ALTO RISCO — recomenda-se revisão de código e testes adicionais'
        ELSE 'BAIXO RISCO — pode seguir o fluxo normal'
    END                                                           AS recomendacao
FROM exps
ORDER BY caso_id;
