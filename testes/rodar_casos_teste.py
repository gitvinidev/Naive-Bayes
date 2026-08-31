#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Etapa 4 — Roda os 6 casos de teste formais pelo classificador da Etapa 3 e
produz a evidência para o relatório:

  1. (re)constrói dados/classificador.db a partir do CSV da Etapa 2 e executa
     sql/classificador_naive_bayes.sql  (reaproveita as funções da Etapa 3);
  2. carrega testes/casos_teste.csv na tabela caso_teste (formato longo);
  3. consulta a view classificar_modulo para os 6 casos;
  4. calcula, a partir da view `verossimilhancas`, o log-odds de cada
     categoria de cada feature  ->  ln( P(cat|SIM) / P(cat|NAO) )
     e o ranking das 6 features por |log-odds| médio (poder discriminativo);
  5. imprime tudo e grava em testes/resultados_casos.txt.

Não contém análise escrita — essa está em relatorios/etapa4_resultados.md.
"""

import csv
import sqlite3
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "sql"))
# funções já escritas e testadas na Etapa 3
from rodar_classificador import garantir_math, importar_treino, FEATURES, SQL_SCRIPT  # noqa: E402

DB_PATH = RAIZ / "dados" / "classificador.db"
CASOS_CSV = RAIZ / "testes" / "casos_teste.csv"
SAIDA_TXT = RAIZ / "testes" / "resultados_casos.txt"


def carregar_casos_csv(con):
    """Lê o CSV largo e insere cada caso como 6 linhas (caso_id, feature, categoria)."""
    con.execute("DELETE FROM caso_teste;")
    casos = {}
    with open(CASOS_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cid = row["caso_id"]
            casos[cid] = {
                "proposito": row["proposito"],
                "intuicao": row["intuicao_esperada"],
                "perfil": {f: row[f] for f in FEATURES},
            }
            for f in FEATURES:
                con.execute(
                    "INSERT INTO caso_teste (caso_id, feature, categoria) VALUES (?,?,?)",
                    (cid, f, row[f]),
                )
    con.commit()
    return casos


def combinacao_vista_no_treino(con, perfil):
    """Conta quantos módulos de treino têm EXATAMENTE este perfil de 6 categorias."""
    sql = """
        SELECT COUNT(*) FROM dados_treinamento
        WHERE complexidade_cat = :complexidade
          AND loc_cat          = :loc
          AND n_autores_cat    = :n_autores
          AND churn_relativo_cat = :churn
          AND n_imports_cat    = :n_imports
          AND cobertura_testes_cat = :cobertura
    """
    return con.execute(sql, perfil).fetchone()[0]


def classificar(con, casos):
    linhas = con.execute(
        """
        SELECT caso_id, n_features, log_score_sim, log_score_nao,
               prob_sim_pct, prob_nao_pct, recomendacao
        FROM classificar_modulo ORDER BY caso_id
        """
    ).fetchall()
    out = []
    for cid, n_feat, lsim, lnao, psim, pnao, rec in linhas:
        info = casos[cid]
        vistos = combinacao_vista_no_treino(con, info["perfil"])
        out.append({
            "caso_id": cid, "proposito": info["proposito"],
            "perfil": info["perfil"], "intuicao": info["intuicao"],
            "n_features": n_feat, "log_sim": lsim, "log_nao": lnao,
            "prob_sim": psim, "prob_nao": pnao, "recomendacao": rec,
            "perfil_exato_no_treino": vistos,
        })
    return out


def log_odds(con):
    """Log-odds por (feature, categoria) e ranking das features por |log-odds| médio."""
    detalhe = con.execute(
        """
        SELECT s.feature, s.categoria,
               ROUND(s.p_verossimilhanca, 4)                       AS p_sim,
               ROUND(n.p_verossimilhanca, 4)                       AS p_nao,
               ROUND(ln(s.p_verossimilhanca / n.p_verossimilhanca), 4) AS log_odds
        FROM verossimilhancas s
        JOIN verossimilhancas n
          ON n.feature = s.feature AND n.categoria = s.categoria AND n.classe = 'NAO'
        WHERE s.classe = 'SIM'
        ORDER BY s.feature, s.categoria
        """
    ).fetchall()

    ranking = con.execute(
        """
        WITH lo AS (
            SELECT s.feature,
                   ln(s.p_verossimilhanca / n.p_verossimilhanca) AS log_odds
            FROM verossimilhancas s
            JOIN verossimilhancas n
              ON n.feature = s.feature AND n.categoria = s.categoria AND n.classe = 'NAO'
            WHERE s.classe = 'SIM'
        )
        SELECT feature,
               ROUND(AVG(ABS(log_odds)), 4) AS logodds_abs_medio,
               ROUND(MAX(ABS(log_odds)), 4) AS logodds_abs_max
        FROM lo
        GROUP BY feature
        ORDER BY logodds_abs_medio DESC
        """
    ).fetchall()
    return detalhe, ranking


def formatar(casos_result, detalhe, ranking):
    L = []
    L.append("=" * 72)
    L.append("ETAPA 4 — RESULTADOS DOS 6 CASOS DE TESTE + RANKING DE LOG-ODDS")
    L.append("=" * 72)

    for r in casos_result:
        L.append("")
        L.append(f"[{r['caso_id']}]  {r['proposito']}")
        L.append("  perfil ............: "
                 + ", ".join(f"{k}={v}" for k, v in r["perfil"].items()))
        L.append(f"  features casadas ..: {r['n_features']}/6")
        L.append(f"  perfil exato visto no treino .: "
                 f"{r['perfil_exato_no_treino']} modulo(s)")
        L.append(f"  log-score SIM / NAO .........: "
                 f"{r['log_sim']:.4f} / {r['log_nao']:.4f}")
        L.append(f"  P(SIM) / P(NAO) ............: "
                 f"{r['prob_sim']:.2f}% / {r['prob_nao']:.2f}%  "
                 f"(soma {r['prob_sim'] + r['prob_nao']:.2f}%)")
        L.append(f"  recomendacao ..............: {r['recomendacao']}")
        L.append(f"  intuicao esperada .........: {r['intuicao']}")

    L.append("")
    L.append("-" * 72)
    L.append("LOG-ODDS POR CATEGORIA   ln( P(cat|SIM) / P(cat|NAO) )")
    L.append("  ( > 0  => categoria empurra para SIM ;  < 0 => empurra para NAO )")
    L.append("-" * 72)
    L.append(f"  {'feature':<14}{'categoria':<10}{'P(cat|SIM)':>12}{'P(cat|NAO)':>12}{'log-odds':>12}")
    for feat, cat, psim, pnao, lo in detalhe:
        L.append(f"  {feat:<14}{cat:<10}{psim:>12}{pnao:>12}{lo:>12}")

    L.append("")
    L.append("-" * 72)
    L.append("RANKING DAS FEATURES POR PODER DISCRIMINATIVO ( |log-odds| medio )")
    L.append("-" * 72)
    L.append(f"  {'#':<3}{'feature':<16}{'|log-odds| medio':>18}{'|log-odds| max':>16}")
    for i, (feat, med, mx) in enumerate(ranking, 1):
        L.append(f"  {i:<3}{feat:<16}{med:>18}{mx:>16}")
    L.append("")
    L.append("=" * 72)
    return "\n".join(L)


def main():
    con = sqlite3.connect(DB_PATH)
    try:
        garantir_math(con)
        n = importar_treino(con)
        con.executescript(SQL_SCRIPT.read_text(encoding="utf-8"))
        casos = carregar_casos_csv(con)
        casos_result = classificar(con, casos)
        detalhe, ranking = log_odds(con)
    finally:
        con.close()

    texto = formatar(casos_result, detalhe, ranking)
    SAIDA_TXT.write_text(texto + "\n", encoding="utf-8")
    print(f"(base reconstruida com {n} registros de treino)\n")
    print(texto)
    print(f"\nResultados salvos em: {SAIDA_TXT}")

    # validação mínima do smoke test
    for r in casos_result:
        assert r["n_features"] == 6, r
        assert abs(r["prob_sim"] + r["prob_nao"] - 100.0) < 0.05, r


if __name__ == "__main__":
    main()
