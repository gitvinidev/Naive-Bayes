#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Etapa 2 — Geração da massa de dados sintética de treinamento.

Domínio: predição de defeitos de software a partir de métricas estáticas de código.
Unidade de análise: arquivo ("módulo").

O script NÃO coleta dados de repositórios reais: ele os SINTETIZA. Isso é
exigência explícita da atividade (Etapa 2 do enunciado).

DECISÃO DE PROJETO — "Naive Bayes puro" (ver CLAUDE.md):
  As 6 features são sorteadas de suas próprias distribuições, cada uma
  INDEPENDENTE das demais — nenhuma feature depende do valor de outra feature
  neste gerador (sem tendência cruzada, sem ruído compartilhado, sem
  subpopulação de interação). Só o RÓTULO (defeito SIM/NÃO) depende das 6
  features — isso é esperado e correto: é literalmente o que o Naive Bayes
  modela, P(feature | classe).

  Por quê: isso testa o classificador no cenário em que sua própria suposição
  central (independência condicional entre features) é verdadeira POR
  CONSTRUÇÃO dos dados — o "Naive Bayes em sua forma pura". É uma escolha de
  simulação legítima, declarada como tal: a literatura (Shepperd, 1988;
  R² ≈ 0,93 entre complexidade e LOC) documenta que, no mundo real, essas
  features SERIAM correlacionadas. Nossos dados sintéticos optam por não
  reproduzir essa correlação, de propósito, para isolar o comportamento
  teórico do modelo (discussão completa nos relatórios das Etapas 1 e 4).

  NÃO fazer: não reintroduzir nenhuma dependência entre features aqui.

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
#    Cada feature é sorteada de sua PRÓPRIA distribuição, com seu PRÓPRIO
#    gerador de números aleatórios (ver rng.spawn() no main()) — nenhuma
#    lê o valor de outra. Mesmas famílias de distribuição de antes
#    (log-normal para tamanho/contagens assimétricas, Poisson para
#    contagens de pessoas, normal para percentuais), só sem acoplamento.
# ==========================================================================
# --- LOC: distribuição log-normal (muitos arquivos pequenos, poucos enormes) ---
LOC_LOG_MEDIA = 4.5        # e^4.5 ~ 90 linhas na mediana
LOC_LOG_SIGMA = 0.95
LOC_MIN, LOC_MAX = 8, 1500

# --- Complexidade ciclomática: log-normal própria, independente de LOC ---
CC_LOG_MEDIA = 2.15        # e^2.15 ~ 8,6 na mediana (perto do corte "baixo <= 10")
CC_LOG_SIGMA = 0.60
FRACAO_MODULOS_DENSOS = 0.12   # fração de arquivos com complexidade desproporcional
CC_MULT_DENSO = (2.2, 3.8)     # multiplicador extra de complexidade nesses arquivos
                                # (sorteio independente — não olha LOC nem nenhuma outra feature)
CC_MIN, CC_MAX = 1, 120

# --- Nº de autores distintos: Poisson própria, independente do tamanho ---
AUT_LAMBDA = 1.55           # média fixa — não escala mais com LOC
AUT_MIN, AUT_MAX = 1, 9

# --- Churn relativo (% do arquivo alterado nos últimos 90 dias): TAXA, própria ---
CHURN_LOG_MEDIA = 2.35    # e^2.35 ~ 10,5% na mediana
CHURN_LOG_SIGMA = 0.75
CHURN_MIN, CHURN_MAX = 0.3, 85.0

# --- Nº de imports/dependências externas: própria, independente de LOC ---
IMP_BASE = 3.2             # nível médio fixo — não escala mais com LOC
IMP_RUIDO_SIGMA = 0.90     # variação própria (ruído log-normal multiplicativo)
IMP_POISSON = 3.0          # componente aditivo (contagem "de base")
IMP_MIN, IMP_MAX = 0, 40

# --- Cobertura de testes (%): própria, independente do churn ---
COB_MEDIA, COB_SIGMA = 63.0, 19.0
COB_MIN, COB_MAX = 2.0, 99.0

# ==========================================================================
# 3) PESOS DO MODELO DE RÓTULO  (log-odds de defeito por categoria de cada feature)
#    Sinais coerentes com a literatura; ver justificativa no relatório da Etapa 1.
#    O RÓTULO é a única coisa que depende das features — é isso que o Naive
#    Bayes modela (P(feature | classe)); não é uma dependência entre features.
# ==========================================================================
PESOS = {
    # LOC: 'baixo' pesa MAIS que 'alto' — módulos menores são proporcionalmente
    # mais propensos a defeito (Koru et al.); 'alto' ainda soma risco por volume.
    # Efeito em U de propósito — não é um erro de discretização.
    "loc":       {"baixo": 0.45, "medio": 0.00, "alto": 0.20},
    "cc":        {"baixo": -0.35, "medio": 0.25, "alto": 0.75},
    "n_autores": {"baixo": -0.35, "medio": 0.10, "alto": 0.90},
    "churn":     {"baixo": -0.40, "medio": 0.15, "alto": 0.85},
    "n_imports": {"baixo": -0.15, "medio": 0.05, "alto": 0.35},
    # Cobertura: efeito pequeno e no limite do ruído — feature contestada.
    "cobertura": {"baixo": 0.20, "medio": 0.00, "alto": -0.15},
}
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
def gerar_features(rngs):
    """rngs: dict com um np.random.Generator INDEPENDENTE por feature (ver
    main()) — garante que nenhum sorteio compartilha ruído com outro."""
    n = N_REGISTROS

    # --- LOC ---
    loc = rngs["loc"].lognormal(LOC_LOG_MEDIA, LOC_LOG_SIGMA, size=n)
    loc = np.clip(np.round(loc), LOC_MIN, LOC_MAX).astype(int)

    # --- Complexidade ciclomática: log-normal própria (NÃO lê `loc`) ---
    cc = rngs["cc"].lognormal(CC_LOG_MEDIA, CC_LOG_SIGMA, size=n)
    # subpopulação aleatória com complexidade acima da tendência (ruído
    # próprio, sorteada com o rng da própria feature — independente de LOC
    # e de qualquer outra coluna)
    denso = rngs["cc"].random(n) < FRACAO_MODULOS_DENSOS
    cc[denso] *= rngs["cc"].uniform(*CC_MULT_DENSO, size=int(denso.sum()))
    cc = np.clip(np.round(cc), CC_MIN, CC_MAX).astype(int)

    # --- Nº de autores distintos: Poisson própria (NÃO lê `loc`) ---
    n_autores = 1 + rngs["n_autores"].poisson(AUT_LAMBDA, size=n)
    n_autores = np.clip(n_autores, AUT_MIN, AUT_MAX).astype(int)

    # --- Churn relativo: TAXA (% do arquivo em 90 dias), própria ---
    churn = rngs["churn"].lognormal(CHURN_LOG_MEDIA, CHURN_LOG_SIGMA, size=n)
    churn = np.clip(np.round(churn, 1), CHURN_MIN, CHURN_MAX)

    # --- Nº de imports: própria (NÃO lê `loc`) ---
    n_imports = (IMP_BASE * rngs["n_imports"].lognormal(0.0, IMP_RUIDO_SIGMA, size=n)
                 + rngs["n_imports"].poisson(IMP_POISSON, size=n))
    n_imports = np.clip(np.round(n_imports), IMP_MIN, IMP_MAX).astype(int)

    # --- Cobertura de testes: própria (NÃO lê `churn`) ---
    cobertura = rngs["cobertura"].normal(COB_MEDIA, COB_SIGMA, size=n)
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
def gerar_rotulo(df, rng_rotulo):
    """O rótulo é a ÚNICA coisa que depende das 6 features — soma de pesos
    (log-odds) por categoria + ruído. Sem bônus de interação entre features:
    cada uma contribui de forma independente das demais (ver Seção 1 do
    módulo e CLAUDE.md, "Naive Bayes puro")."""
    score = np.zeros(len(df))
    score += df["loc_cat"].map(PESOS["loc"]).to_numpy()
    score += df["complexidade_cat"].map(PESOS["cc"]).to_numpy()
    score += df["n_autores_cat"].map(PESOS["n_autores"]).to_numpy()
    score += df["churn_relativo_cat"].map(PESOS["churn"]).to_numpy()
    score += df["n_imports_cat"].map(PESOS["n_imports"]).to_numpy()
    score += df["cobertura_testes_cat"].map(PESOS["cobertura"]).to_numpy()

    # ruído: impede que o rótulo seja função determinística das categorias
    score += rng_rotulo.normal(0.0, SIGMA_RUIDO_ROTULO, size=len(df))

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
    # Resultado ESPERADO: todos os |r| perto de 0. As 6 features são sorteadas
    # de forma independente (Seção "Naive Bayes puro" no topo do arquivo) —
    # esta matriz confirma isso, não "confirma padrões programados".
    corr = df[NUM_COLS].corr(method="pearson").round(3)
    L.append("(b) MATRIZ DE CORRELAÇÃO DE PEARSON (features numéricas)")
    L.append("    Esperado: todos os |r| próximos de 0 — as 6 features são")
    L.append("    geradas independentes entre si por decisão de projeto")
    L.append("    (\"Naive Bayes puro\", ver CLAUDE.md). Valores não-nulos aqui")
    L.append("    são ruído amostral (N=150), não dependência estrutural.")
    L.append(corr.to_string())
    L.append("")
    maior_par, maior_val = None, 0.0
    for i, a in enumerate(NUM_COLS):
        for b in NUM_COLS[i + 1:]:
            v = abs(corr.loc[a, b])
            if v > maior_val:
                maior_val, maior_par = v, (a, b)
    L.append(f"    -> Maior |r| da matriz: {ROTULOS_NUM[maior_par[0]]} x "
              f"{ROTULOS_NUM[maior_par[1]]} = {corr.loc[maior_par]:.3f}")
    L.append("")

    # --- (c) estatísticas descritivas básicas ---
    desc = df[NUM_COLS].agg(["mean", "std", "min", "median", "max"]).T
    desc = desc.round(2)
    L.append("(c) ESTATÍSTICAS DESCRITIVAS (valor bruto de cada feature)")
    L.append(desc.to_string())
    L.append("")

    # --- (d) taxa de defeito por categoria de cada feature (isoladamente) ---
    # O rótulo AINDA depende de cada feature individualmente (é o que o Naive
    # Bayes modela) — só não há mais dependência ENTRE features. Por isso essa
    # checagem, feature a feature, continua fazendo sentido.
    taxa_geral = (df["defeito"] == "SIM").mean()
    L.append("(d) TAXA DE DEFEITO POR CATEGORIA, FEATURE A FEATURE  (P(SIM))")
    L.append("    (cada feature isolada ainda influencia o rótulo por construção;")
    L.append("     o que deixou de existir é dependência ENTRE as features)")
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
    L.append("")
    L.append("    Esperado: crescente com a categoria para complexidade, autores,")
    L.append("    churn e imports; em U para LOC (Koru et al., 2008 — baixo E alto")
    L.append("    mais arriscados que médio); fraco/quase plano para cobertura")
    L.append("    (feature de evidência contestada) — ver pesos no topo do script.")
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
                   "descritivas": desc, "taxa_geral": taxa_geral}


# ==========================================================================
# 8) MAIN
# ==========================================================================
def main():
    rng = np.random.default_rng(SEED)

    # Um gerador INDEPENDENTE por feature + um para o ruído do rótulo,
    # todos derivados (deterministicamente) da mesma SEED via spawn(): isso
    # garante que nenhuma feature compartilha ruído com outra — condição
    # necessária para a decisão "Naive Bayes puro" (ver docstring do módulo).
    chaves = ["loc", "cc", "n_autores", "churn", "n_imports", "cobertura", "rotulo"]
    filhos = rng.spawn(len(chaves))
    rngs = dict(zip(chaves, filhos))

    df = gerar_features(rngs)
    df = gerar_rotulo(df, rngs["rotulo"])

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
