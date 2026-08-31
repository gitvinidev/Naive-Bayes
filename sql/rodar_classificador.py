#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Etapa 3 — Runner do classificador Naive Bayes em SQL (SQLite).

O que este script faz, em ordem:
  1. (Re)cria o banco dados/classificador.db e importa
     dados/etapa2_dados_treinamento.csv para a tabela `dados_treinamento`.
  2. Garante que as funções LN e EXP existem no SQLite (registra
     equivalentes em Python se o build não as tiver — ver nota abaixo).
  3. Executa sql/classificador_naive_bayes.sql (cria as views do modelo).
  4. Insere 1–2 CASOS DE EXEMPLO MÍNIMOS em `caso_teste` — apenas um
     smoke test para provar que o SQL roda de ponta a ponta. Os 5+ casos
     formais e a análise crítica são da Etapa 4, não deste script.
  5. Consulta a view `classificar_modulo` e imprime o resultado legível.
  6. Valida: as duas probabilidades somam ~100% e as 6 features casaram.

NOTA sobre LN/EXP:
  O SQLite >= 3.35 tem ln()/exp() nativas quando compilado com
  SQLITE_ENABLE_MATH_FUNCTIONS. O módulo sqlite3 do CPython usado aqui
  (SQLite 3.45.1) já as traz. Mesmo assim, por portabilidade, testamos
  `SELECT ln(1)` e, se falhar, registramos `ln` e `exp` via
  connection.create_function(..., math.log / math.exp). Assim o mesmo
  SQL roda em qualquer build.
"""

import csv
import math
import sqlite3
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
CSV_TREINO = RAIZ / "dados" / "etapa2_dados_treinamento.csv"
DB_PATH = RAIZ / "dados" / "classificador.db"
SQL_SCRIPT = RAIZ / "sql" / "classificador_naive_bayes.sql"

# Esquema da tabela de treino (nomes = cabeçalho do CSV da Etapa 2)
COLUNAS_TREINO = [
    ("modulo_id", "TEXT"),
    ("complexidade_ciclomatica", "INTEGER"),
    ("complexidade_cat", "TEXT"),
    ("loc", "INTEGER"),
    ("loc_cat", "TEXT"),
    ("n_autores", "INTEGER"),
    ("n_autores_cat", "TEXT"),
    ("churn_relativo", "REAL"),
    ("churn_relativo_cat", "TEXT"),
    ("n_imports", "INTEGER"),
    ("n_imports_cat", "TEXT"),
    ("cobertura_testes", "REAL"),
    ("cobertura_testes_cat", "TEXT"),
    ("defeito", "TEXT"),
]

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


def garantir_math(con):
    """Testa ln()/exp(); registra fallback Python se o build não tiver."""
    try:
        con.execute("SELECT ln(1.0), exp(0.0)").fetchone()
        return "nativas do SQLite"
    except sqlite3.OperationalError:
        con.create_function("ln", 1, math.log, deterministic=True)
        con.create_function("exp", 1, math.exp, deterministic=True)
        return "registradas em Python (math.log / math.exp)"


def importar_treino(con):
    """(Re)cria dados_treinamento e carrega o CSV da Etapa 2."""
    cols_ddl = ",\n    ".join(f"{nome} {tipo}" for nome, tipo in COLUNAS_TREINO)
    con.execute("DROP TABLE IF EXISTS dados_treinamento;")
    con.execute(f"CREATE TABLE dados_treinamento (\n    {cols_ddl}\n);")

    nomes = [nome for nome, _ in COLUNAS_TREINO]
    placeholders = ",".join("?" for _ in nomes)
    with open(CSV_TREINO, newline="", encoding="utf-8") as fh:
        leitor = csv.DictReader(fh)
        faltando = set(nomes) - set(leitor.fieldnames or [])
        if faltando:
            raise SystemExit(f"CSV sem colunas esperadas: {sorted(faltando)}")
        linhas = [tuple(row[n] for n in nomes) for row in leitor]
    con.executemany(
        f"INSERT INTO dados_treinamento ({','.join(nomes)}) VALUES ({placeholders})",
        linhas,
    )
    con.commit()
    return len(linhas)


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
    con.commit()


def imprimir_priors(con):
    print("PRIORS  P(classe)  (contagem simples em dados_treinamento)")
    for classe, n, total, p in con.execute(
        "SELECT classe, n_classe, n_total, p_prior FROM priors ORDER BY classe"
    ):
        print(f"   P({classe}) = {n:>3}/{total} = {p:.4f}")
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
    con = sqlite3.connect(DB_PATH)
    try:
        origem_math = garantir_math(con)
        print(f"LN/EXP: {origem_math}\n")

        n = importar_treino(con)
        print(f"Importados {n} registros para dados_treinamento.\n")

        con.executescript(SQL_SCRIPT.read_text(encoding="utf-8"))
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
