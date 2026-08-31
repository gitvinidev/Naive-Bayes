#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Etapa 2 — Geração da massa de dados sintética de treinamento.

Domínio: predição de defeitos de software a partir de métricas estáticas de código.
Unidade de análise: arquivo ("módulo").

O script NÃO coleta dados de repositórios reais: ele os SINTETIZA, programando
deliberadamente os padrões levantados na literatura (ver CLAUDE.md e o relatório
da Etapa 1). Isso é exigência explícita da atividade e serve de base para a
análise crítica da Etapa 4.

Padrões programados de propósito:
  - Complexidade ciclomática e LOC nascem correlacionadas          (Shepperd, 1988)
  - Módulo pequeno + denso (LOC baixo + complexidade alta) = risco  (Koru et al., 2008)
  - Churn é uma TAXA RELATIVA (% do arquivo alterado), não absoluta  (Nagappan & Ball, 2005)
  - Nº de imports tem correlação LEVE com LOC, mas variação própria  (analogia a CBO x tamanho)
  - Cobertura de testes é sinal FRACO/contestado                     (Inozemtseva & Holmes, 2014;
                                                                      Gren & Antinyan, 2017)
  - Ruído controlado em todas as features: nada de sorteio uniforme
    puro nem regra determinística perfeita.

Saídas:
  dados/etapa2_dados_treinamento.csv   — 6 features (valor bruto + categoria) + rótulo
  dados/etapa2_validacao.txt           — proporção de classes, correlações, estatísticas
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ==========================================================================
# 1) PARÂMETROS PRINCIPAIS  (ajuste aqui — nada de valor solto no meio do código)
# ==========================================================================
N_REGISTROS = 150          # nº de módulos gerados (mínimo exigido pela atividade: 100)
SEED = 42                  # semente única — usada em TODO gerador aleatório do script
PROPORCAO_DEFEITO = 0.35   # fração de módulos com defeito=SIM (longe de 0,5 e dos extremos)

# ==========================================================================
# 2) PARÂMETROS DO MODELO GENERATIVO DAS FEATURES
# ==========================================================================
# --- LOC: distribuição log-normal (muitos arquivos pequenos, poucos enormes) ---
LOC_LOG_MEDIA = 4.5        # e^4.5 ~ 90 linhas na mediana
LOC_LOG_SIGMA = 0.95
LOC_MIN, LOC_MAX = 8, 1500

# --- Complexidade ciclomática: tendência crescente com LOC + ruído próprio ---
CC_BASE = 1.0             # complexidade mínima esperada (1 caminho)
CC_POR_LOC = 0.085        # inclinação da tendência linear sobre LOC
CC_RUIDO_SIGMA = 0.45     # ruído log-normal multiplicativo (variação própria)
FRACAO_MODULOS_DENSOS = 0.12   # fração de arquivos com complexidade desproporcional
CC_MULT_DENSO = (2.2, 3.8)     # multiplicador extra de complexidade nesses arquivos
# subpopulação FORÇADA "pequeno e denso" (arquivo curto + complexidade alta):
# representa explicitamente o achado de Koru et al. — pequeno e denso = arriscado.
LOC_LIMITE_PEQUENO = 55        # limite de LOC para ser considerado "pequeno" aqui
FRACAO_PEQUENO_DENSO = 0.06    # ~9 módulos em 150
CC_FORCADO_PEQUENO_DENSO = (22, 46)  # complexidade cravada na faixa "alto"
CC_MIN, CC_MAX = 1, 120

# --- Nº de autores distintos: ligação LEVE com o tamanho ---
AUT_LAMBDA_BASE = 1.15
AUT_LAMBDA_POR_LOC = 1 / 450
AUT_MIN, AUT_MAX = 1, 9

# --- Churn relativo (% do arquivo alterado nos últimos 90 dias): TAXA, não absoluto ---
CHURN_LOG_MEDIA = 2.35    # e^2.35 ~ 10,5% na mediana
CHURN_LOG_SIGMA = 0.75
CHURN_MIN, CHURN_MAX = 0.3, 85.0

# --- Nº de imports/dependências externas: correlação LEVE com LOC + variação própria ---
IMP_BASE = 1.5
IMP_POR_LOC = 1 / 200     # ligação fraca com o tamanho (coef. pequeno de propósito)
IMP_RUIDO_SIGMA = 0.90    # ruído próprio forte
IMP_POISSON = 3.0         # componente aditivo independente do tamanho
IMP_MIN, IMP_MAX = 0, 40

# --- Cobertura de testes (%): quase independente; leve queda quando o churn é alto ---
COB_MEDIA, COB_SIGMA = 63.0, 19.0
COB_EFEITO_CHURN = -0.15  # cada ponto de churn acima de 12% reduz ~0,15 p.p. de cobertura
COB_CHURN_REF = 12.0
COB_MIN, COB_MAX = 2.0, 99.0

# ==========================================================================
# 3) PESOS DO MODELO DE RÓTULO  (log-odds de defeito por categoria de cada feature)
#    Sinais coerentes com a literatura; ver justificativa no relatório da Etapa 1.
# ==========================================================================
PESOS = {
    # LOC: 'baixo' pesa MAIS que 'alto' — módulos menores são proporcionalmente
    # mais propensos a defeito (Koru et al.); 'alto' ainda soma risco por volume.
    "loc":       {"baixo": 0.45, "medio": 0.00, "alto": 0.20},
    "cc":        {"baixo": -0.35, "medio": 0.25, "alto": 0.75},
    "n_autores": {"baixo": -0.35, "medio": 0.10, "alto": 0.90},
    "churn":     {"baixo": -0.40, "medio": 0.15, "alto": 0.85},
    "n_imports": {"baixo": -0.15, "medio": 0.05, "alto": 0.35},
    # Cobertura: efeito pequeno e no limite do ruído — feature contestada.
    "cobertura": {"baixo": 0.20, "medio": 0.00, "alto": -0.15},
}
BONUS_PEQUENO_DENSO = 0.70          # LOC baixo + complexidade alta (o caso de Koru)
BONUS_PEQUENO_DENSO_PARCIAL = 0.30  # LOC baixo + complexidade média
SIGMA_RUIDO_ROTULO = 1.05          # ruído no score => rótulo não é regra determinística

# ==========================================================================
# 4) CAMINHOS E DISCRETIZAÇÃO  (faixas idênticas às do CLAUDE.md / Etapa 1)
# ==========================================================================
DADOS_DIR = Path(__file__).resolve().parent
CSV_OUT = DADOS_DIR / "etapa2_dados_treinamento.csv"
VALID_OUT = DADOS_DIR / "etapa2_validacao.txt"

NUM_COLS = [
    "complexidade_ciclomatica", "loc", "n_autores",
    "churn_relativo", "n_imports", "cobertura_testes",
]
ROTULOS_NUM = {
    "complexidade_ciclomatica": "Complexidade ciclomática",
    "loc": "LOC",
    "n_autores": "Nº de autores",
    "churn_relativo": "Churn relativo (%)",
    "n_imports": "Nº de imports",
    "cobertura_testes": "Cobertura de testes (%)",
}


def cat_complexidade(v):
    return "baixo" if v <= 10 else "medio" if v <= 20 else "alto"


def cat_loc(v):
    return "baixo" if v < 50 else "medio" if v <= 200 else "alto"


def cat_autores(v):
    return "baixo" if v == 1 else "medio" if v <= 3 else "alto"


def cat_churn(v):
    return "baixo" if v < 10 else "medio" if v <= 30 else "alto"


def cat_imports(v):
    return "baixo" if v <= 3 else "medio" if v <= 8 else "alto"


def cat_cobertura(v):
    return "baixo" if v < 50 else "medio" if v <= 80 else "alto"


# ==========================================================================
# 5) GERAÇÃO DAS FEATURES
# ==========================================================================
def gerar_features(rng):
    n = N_REGISTROS

    # --- LOC ---
    loc = rng.lognormal(LOC_LOG_MEDIA, LOC_LOG_SIGMA, size=n)
    loc = np.clip(np.round(loc), LOC_MIN, LOC_MAX).astype(int)

    # --- Complexidade ciclomática: correlacionada com LOC (Shepperd) ---
    cc_esperado = CC_BASE + CC_POR_LOC * loc
    cc = cc_esperado * rng.lognormal(0.0, CC_RUIDO_SIGMA, size=n)
    # (i) subpopulação aleatória com complexidade acima da tendência
    denso = rng.random(n) < FRACAO_MODULOS_DENSOS
    cc[denso] *= rng.uniform(*CC_MULT_DENSO, size=int(denso.sum()))
    # (ii) subpopulação FORÇADA "pequeno e denso": arquivos curtos com
    #      complexidade cravada na faixa "alto" — representa o achado de Koru et al.
    pequenos = np.where(loc < LOC_LIMITE_PEQUENO)[0]
    n_forcar = min(len(pequenos), round(FRACAO_PEQUENO_DENSO * n))
    if n_forcar > 0:
        alvo = rng.choice(pequenos, size=n_forcar, replace=False)
        cc[alvo] = rng.integers(*CC_FORCADO_PEQUENO_DENSO, size=n_forcar)
    cc = np.clip(np.round(cc), CC_MIN, CC_MAX).astype(int)

    # --- Nº de autores distintos: ligação leve com o tamanho ---
    lam_aut = AUT_LAMBDA_BASE + AUT_LAMBDA_POR_LOC * loc
    n_autores = 1 + rng.poisson(lam_aut, size=n)
    n_autores = np.clip(n_autores, AUT_MIN, AUT_MAX).astype(int)

    # --- Churn relativo: TAXA (% do arquivo em 90 dias), independente do tamanho absoluto ---
    churn = rng.lognormal(CHURN_LOG_MEDIA, CHURN_LOG_SIGMA, size=n)
    churn = np.clip(np.round(churn, 1), CHURN_MIN, CHURN_MAX)

    # --- Nº de imports: correlação leve com LOC + forte variação própria ---
    imp_esperado = IMP_BASE + IMP_POR_LOC * loc
    n_imports = (imp_esperado * rng.lognormal(0.0, IMP_RUIDO_SIGMA, size=n)
                 + rng.poisson(IMP_POISSON, size=n))
    n_imports = np.clip(np.round(n_imports), IMP_MIN, IMP_MAX).astype(int)

    # --- Cobertura de testes: quase independente; leve queda com churn alto ---
    cobertura = (rng.normal(COB_MEDIA, COB_SIGMA, size=n)
                 + COB_EFEITO_CHURN * (churn - COB_CHURN_REF))
    cobertura = np.clip(np.round(cobertura, 1), COB_MIN, COB_MAX)

    df = pd.DataFrame({
        "complexidade_ciclomatica": cc,
        "loc": loc,
        "n_autores": n_autores,
        "churn_relativo": churn,
        "n_imports": n_imports,
        "cobertura_testes": cobertura,
    })
    df["complexidade_cat"] = df["complexidade_ciclomatica"].map(cat_complexidade)
    df["loc_cat"] = df["loc"].map(cat_loc)
    df["n_autores_cat"] = df["n_autores"].map(cat_autores)
    df["churn_relativo_cat"] = df["churn_relativo"].map(cat_churn)
    df["n_imports_cat"] = df["n_imports"].map(cat_imports)
    df["cobertura_testes_cat"] = df["cobertura_testes"].map(cat_cobertura)
    return df


# ==========================================================================
# 6) GERAÇÃO DO RÓTULO (defeito SIM/NÃO)
# ==========================================================================
def gerar_rotulo(df, rng):
    score = np.zeros(len(df))
    score += df["loc_cat"].map(PESOS["loc"]).to_numpy()
    score += df["complexidade_cat"].map(PESOS["cc"]).to_numpy()
    score += df["n_autores_cat"].map(PESOS["n_autores"]).to_numpy()
    score += df["churn_relativo_cat"].map(PESOS["churn"]).to_numpy()
    score += df["n_imports_cat"].map(PESOS["n_imports"]).to_numpy()
    score += df["cobertura_testes_cat"].map(PESOS["cobertura"]).to_numpy()

    # interação pequeno-e-denso (Koru et al.)
    loc_b = df["loc_cat"].to_numpy() == "baixo"
    cc_a = df["complexidade_cat"].to_numpy() == "alto"
    cc_m = df["complexidade_cat"].to_numpy() == "medio"
    score += np.where(loc_b & cc_a, BONUS_PEQUENO_DENSO, 0.0)
    score += np.where(loc_b & cc_m, BONUS_PEQUENO_DENSO_PARCIAL, 0.0)

    # ruído: impede que o rótulo seja função determinística das categorias
    score += rng.normal(0.0, SIGMA_RUIDO_ROTULO, size=len(df))

    # rótulo por limiar top-k => proporção de SIM = PROPORCAO_DEFEITO exatamente
    k = round(len(df) * PROPORCAO_DEFEITO)
    idx_sim = np.argsort(-score)[:k]
    defeito = np.full(len(df), "NAO", dtype=object)
    defeito[idx_sim] = "SIM"

    out = df.copy()
    out["defeito"] = defeito
    return out


# ==========================================================================
# 7) VALIDAÇÃO
# ==========================================================================
def validar(df):
    """Gera o texto de validação e devolve também os objetos usados no relatório."""
    L = []
    L.append("=" * 70)
    L.append("VALIDAÇÃO DA MASSA DE DADOS SINTÉTICA — ETAPA 2")
    L.append("=" * 70)
    L.append(f"Parâmetros: N_REGISTROS={N_REGISTROS}  SEED={SEED}  "
             f"PROPORCAO_DEFEITO={PROPORCAO_DEFEITO}")
    L.append("")

    # --- (a) proporção de classes ---
    cont = df["defeito"].value_counts().reindex(["SIM", "NAO"]).fillna(0).astype(int)
    prop = df["defeito"].value_counts(normalize=True).reindex(["SIM", "NAO"]).fillna(0)
    L.append("(a) PROPORÇÃO DE CLASSES")
    L.append(f"    SIM : {cont['SIM']:3d}  ({prop['SIM']*100:5.1f}%)")
    L.append(f"    NAO : {cont['NAO']:3d}  ({prop['NAO']*100:5.1f}%)")
    L.append("")

    # --- (b) matriz de correlação (Pearson) entre as 6 features numéricas ---
    corr = df[NUM_COLS].corr(method="pearson").round(3)
    L.append("(b) MATRIZ DE CORRELAÇÃO DE PEARSON (features numéricas)")
    L.append(corr.to_string())
    L.append("")
    L.append(f"    -> Correlação LOC x complexidade ciclomática = "
             f"{corr.loc['loc', 'complexidade_ciclomatica']:.3f}  "
             f"(programada de propósito — Shepperd, 1988)")
    L.append(f"    -> Correlação LOC x nº de imports            = "
             f"{corr.loc['loc', 'n_imports']:.3f}  (esperada: leve)")
    L.append(f"    -> Correlação LOC x churn relativo           = "
             f"{corr.loc['loc', 'churn_relativo']:.3f}  (esperada: ~0 — churn é taxa)")
    L.append("")

    # --- (c) estatísticas descritivas básicas ---
    desc = df[NUM_COLS].agg(["mean", "std", "min", "median", "max"]).T
    desc = desc.round(2)
    L.append("(c) ESTATÍSTICAS DESCRITIVAS (valor bruto de cada feature)")
    L.append(desc.to_string())
    L.append("")

    # --- (d) checagem dos padrões programados ---
    taxa_geral = (df["defeito"] == "SIM").mean()
    L.append("(d) CHECAGEM DOS PADRÕES PROGRAMADOS  (taxa de defeito = P(SIM))")
    L.append(f"    Taxa geral de defeito .................... {taxa_geral*100:5.1f}%")

    for col, nome in [("loc_cat", "LOC"), ("complexidade_cat", "Complexidade"),
                      ("churn_relativo_cat", "Churn rel."),
                      ("n_autores_cat", "Nº autores"),
                      ("n_imports_cat", "Nº imports"),
                      ("cobertura_testes_cat", "Cobertura")]:
        tx = df.groupby(col, observed=True)["defeito"].apply(lambda s: (s == "SIM").mean())
        tx = tx.reindex(["baixo", "medio", "alto"])
        partes = "  ".join(
            f"{c}={tx[c]*100:4.0f}%" if pd.notna(tx[c]) else f"{c}=  - "
            for c in ["baixo", "medio", "alto"]
        )
        L.append(f"    {nome:<13}: {partes}")

    mask_pd = (df["loc_cat"] == "baixo") & (df["complexidade_cat"] == "alto")
    n_pd = int(mask_pd.sum())
    tx_pd = (df.loc[mask_pd, "defeito"] == "SIM").mean() if n_pd else float("nan")
    L.append("")
    L.append(f"    Módulos 'pequeno e denso' (LOC baixo + complexidade alta): "
             f"n={n_pd}")
    L.append(f"      taxa de defeito nesse grupo ........... "
             f"{tx_pd*100:5.1f}%   (esperado: bem acima da taxa geral — Koru et al.)")
    L.append("")

    # --- distribuição das categorias ---
    L.append("(e) DISTRIBUIÇÃO DAS CATEGORIAS")
    for col in ["loc_cat", "complexidade_cat", "n_autores_cat",
                "churn_relativo_cat", "n_imports_cat", "cobertura_testes_cat"]:
        d = df[col].value_counts().reindex(["baixo", "medio", "alto"]).fillna(0).astype(int)
        L.append(f"    {col:<22}: baixo={d['baixo']:3d}  medio={d['medio']:3d}  alto={d['alto']:3d}")
    L.append("")
    L.append("=" * 70)

    texto = "\n".join(L)
    return texto, {"contagem": cont, "proporcao": prop, "correlacao": corr,
                   "descritivas": desc, "taxa_pequeno_denso": tx_pd,
                   "n_pequeno_denso": n_pd, "taxa_geral": taxa_geral}


# ==========================================================================
# 8) MAIN
# ==========================================================================
def main():
    rng = np.random.default_rng(SEED)

    df = gerar_features(rng)
    df = gerar_rotulo(df, rng)

    # coluna de ID de módulo + ordenação final das colunas
    df.insert(0, "modulo_id",
              [f"modulo_{i:03d}" for i in range(1, len(df) + 1)])
    ordem = ["modulo_id",
             "complexidade_ciclomatica", "complexidade_cat",
             "loc", "loc_cat",
             "n_autores", "n_autores_cat",
             "churn_relativo", "churn_relativo_cat",
             "n_imports", "n_imports_cat",
             "cobertura_testes", "cobertura_testes_cat",
             "defeito"]
    df = df[ordem]

    df.to_csv(CSV_OUT, index=False, encoding="utf-8")

    texto, _ = validar(df)
    VALID_OUT.write_text(texto + "\n", encoding="utf-8")

    print(texto)
    print(f"\nCSV salvo em .......: {CSV_OUT}")
    print(f"Validação salva em : {VALID_OUT}")


if __name__ == "__main__":
    main()
