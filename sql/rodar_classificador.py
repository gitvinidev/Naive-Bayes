#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Etapa 3 — Runner do classificador Naive Bayes em SQL (DuckDB).

O que este script faz, em ordem:
  1. (Re)cria o banco dados/classificador.db e importa
     dados/etapa2_dados_treinamento.csv para a tabela `dados_treinamento`
     (nativamente, via `read_csv_auto`).
  2. Executa sql/classificador_naive_bayes.sql (cria as views do modelo).
  3. Insere 1–2 CASOS DE EXEMPLO MÍNIMOS em `caso_teste` — apenas um
     smoke test para provar que o SQL roda de ponta a ponta. Os 5+ casos
     formais e a análise crítica são da Etapa 4, não deste script.
  4. Consulta a view `classificar_modulo` e imprime o resultado legível.
  5. Valida: as duas probabilidades somam ~100% e as 6 features casaram.

NOTA sobre LN/EXP:
  DuckDB traz `ln()` e `exp()` nativos — diferente do SQLite, não é preciso
  nenhum fallback em Python (`connection.create_function`) para garantir
  essas funções; ver CLAUDE.md, "Migração para DuckDB".
"""

import duckdb
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
CSV_TREINO = RAIZ / "dados" / "etapa2_dados_treinamento.csv"
DB_PATH = RAIZ / "dados" / "classificador.db"
SQL_SCRIPT = RAIZ / "sql" / "classificador_naive_bayes.sql"

# As 6 chaves de feature usadas pelo SQL (iguais às de treino_longo)
FEATURES = ["complexidade", "loc", "n_autores", "churn", "n_imports", "cobertura"]

# ----------------------------------------------------------------------
# Casos de EXEMPLO (smoke test — NÃO são os casos formais da Etapa 4)
# ----------------------------------------------------------------------
CASOS_EXEMPLO = {
    # arquivo pequeno e denso, muito mexido, mal coberto -> deve dar ALTO RISCO
    "exemplo_alto_risco": {
        "complexidade": "alto",
        "loc": "baixo",
        "n_autores": "alto",
        "churn": "alto",
        "n_imports": "alto",
        "cobertura": "baixo",
    },
    # arquivo de tamanho médio, simples, estável, bem coberto -> deve dar BAIXO RISCO
    "exemplo_baixo_risco": {
        "complexidade": "baixo",
        "loc": "medio",
        "n_autores": "baixo",
        "churn": "baixo",
        "n_imports": "baixo",
        "cobertura": "alto",
    },
}


def importar_treino(con):
    """(Re)cria dados_treinamento a partir do CSV da Etapa 2, via leitura
    nativa do DuckDB — sem laço de INSERT manual."""
    con.execute(f"""
        CREATE OR REPLACE TABLE dados_treinamento AS
        SELECT * FROM read_csv_auto('{CSV_TREINO.as_posix()}');
    """)
    return con.execute("SELECT COUNT(*) FROM dados_treinamento").fetchone()[0]


def carregar_casos(con, casos):
    """Preenche caso_teste (formato longo) com os casos de exemplo."""
    con.execute("DELETE FROM caso_teste;")
    linhas = []
    for caso_id, perfil in casos.items():
        for feat in FEATURES:
            linhas.append((caso_id, feat, perfil[feat]))
    con.executemany(
        "INSERT INTO caso_teste (caso_id, feature, categoria) VALUES (?, ?, ?)",
        linhas,
    )


def imprimir_priors(con):
    print("PRIORS  P(classe)  (contagem simples em dados_treinamento)")
    for classe, n, total, p in con.execute(
        "SELECT classe, n_classe, n_total, p_prior FROM priors ORDER BY classe"
    ).fetchall():
        print(f"   P({classe}) = {n:>7}/{total} = {p:.4f}")
    print()


def classificar_e_imprimir(con, casos):
    linhas = con.execute(
        """
        SELECT caso_id, n_features, log_score_sim, log_score_nao,
               prob_sim_pct, prob_nao_pct, recomendacao
        FROM classificar_modulo
        ORDER BY caso_id
        """
    ).fetchall()

    ok = True
    for (caso_id, n_feat, log_sim, log_nao,
         p_sim, p_nao, recomendacao) in linhas:
        perfil = casos[caso_id]
        print("=" * 66)
        print(f"CASO: {caso_id}")
        print("  perfil:", ", ".join(f"{k}={v}" for k, v in perfil.items()))
        print(f"  features casadas ...: {n_feat}/6")
        print(f"  log-score SIM ......: {log_sim:>10.4f}")
        print(f"  log-score NAO ......: {log_nao:>10.4f}")
        print(f"  P(SIM) ............: {p_sim:6.2f} %")
        print(f"  P(NAO) ............: {p_nao:6.2f} %")
        print(f"  soma ..............: {p_sim + p_nao:6.2f} %")
        print(f"  RECOMENDACAO ......: {recomendacao}")

        # validações do smoke test
        if n_feat != 6:
            print("  !! ERRO: nem todas as 6 features casaram")
            ok = False
        if abs((p_sim + p_nao) - 100.0) > 0.05:
            print("  !! ERRO: probabilidades não somam 100%")
            ok = False
    print("=" * 66)
    return ok


def main():
    print(f"Banco : {DB_PATH}")
    con = duckdb.connect(str(DB_PATH))
    try:
        n = importar_treino(con)
        print(f"Importados {n} registros para dados_treinamento.\n")

        con.execute(SQL_SCRIPT.read_text(encoding="utf-8"))
        carregar_casos(con, CASOS_EXEMPLO)

        imprimir_priors(con)
        ok = classificar_e_imprimir(con, CASOS_EXEMPLO)
    finally:
        con.close()

    if not ok:
        raise SystemExit(1)
    print("\nSmoke test OK: SQL roda de ponta a ponta e as probabilidades "
          "somam 100%.")


if __name__ == "__main__":
    main()
