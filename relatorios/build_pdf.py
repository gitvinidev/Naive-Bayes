#!/usr/bin/env python3
"""Converte um relatorio .md -> HTML estilizado (com cor) -> PDF (via WeasyPrint).

Uso:  python3 relatorios/build_pdf.py [nome_base ...]
      sem argumentos: reconstroi os 4 relatorios da atividade, nesta ordem.
"""
import re
import sys
from pathlib import Path

import markdown
from weasyprint import HTML as WHTML

ROOT = Path(__file__).resolve().parent.parent
RELATORIOS = ROOT / "relatorios"

RODAPE = {
    "etapa1_modelagem": "Etapa 1 — Relatório de Modelagem",
    "etapa2_dados": "Etapa 2 — Massa de Dados de Treinamento",
    "etapa3_classificador": "Etapa 3 — Classificador Naive Bayes em SQL",
    "etapa4_resultados": "Etapa 4 — Resultados dos Testes e Reflexão Crítica",
}

# Paleta: azul-marinho academico + acento dourado
NAVY = "#1f4e79"
NAVY_DARK = "#163a5c"
BLUE = "#2b6ca3"
GOLD = "#c9a227"
TINT = "#eef3f8"     # fundo suave azulado
CODE_BG = "#e8eef4"


def montar_css(base):
    rodape = RODAPE.get(base, base)
    css = f"""
@page {{
  size: A4; margin: 1.8cm 1.9cm;
  @bottom-center {{ content: counter(page) " / " counter(pages);
    font-family: "DejaVu Serif", serif; font-size: 8pt; color: {NAVY}; }}
  @bottom-right {{ content: "{rodape}";
    font-family: "DejaVu Serif", serif; font-size: 7.5pt; color: #9aa7b3; }}
}}
* {{ box-sizing: border-box; }}
body {{
  font-family: "DejaVu Serif", "Liberation Serif", serif;
  font-size: 9.7pt; line-height: 1.38; color: #1a1a1a; margin: 0;
  hyphens: auto;
}}
h1 {{ font-size: 16pt; margin: 0 0 2pt 0; line-height: 1.15; color: {NAVY_DARK};
     border-bottom: 3px solid {GOLD}; padding-bottom: 3pt; }}
h1 + h2 {{ margin-top: 5pt; border: 0; color: {BLUE}; font-size: 11.5pt;
          font-weight: normal; font-style: italic; }}
h2 {{ font-size: 12.5pt; margin: 13pt 0 3pt 0; color: {NAVY};
     border-bottom: 1.5px solid {GOLD}; padding-bottom: 1pt;
     page-break-after: avoid; }}
h3 {{ font-size: 10.3pt; margin: 9pt 0 2pt 0; color: {BLUE};
     page-break-after: avoid; }}
p {{ margin: 3pt 0; text-align: justify; }}
ul, ol {{ margin: 3pt 0; padding-left: 16pt; }}
li {{ margin: 1.5pt 0; }}
li::marker {{ color: {NAVY}; }}
code {{ font-family: "DejaVu Sans Mono", monospace; font-size: 8.4pt;
       background: {CODE_BG}; color: {NAVY_DARK}; padding: 0 2px;
       border-radius: 2px; }}
hr {{ border: 0; border-top: 1.5px solid {GOLD}; margin: 9pt 0; }}
pre {{ background: #f4f6f9; border-left: 3px solid {GOLD}; margin: 6pt 0;
      padding: 5pt 8pt; font-size: 8pt; line-height: 1.3;
      white-space: pre-wrap; overflow-wrap: break-word;
      page-break-inside: avoid; }}
pre code {{ background: none; color: #1a1a1a; padding: 0; }}
strong {{ font-weight: bold; color: {NAVY_DARK}; }}

table {{ border-collapse: collapse; width: 100%; margin: 5pt 0; font-size: 8.4pt;
        line-height: 1.3; page-break-inside: avoid; }}
th, td {{ border: 1px solid #b9c6d2; padding: 2.5pt 4pt; text-align: left;
         vertical-align: top; hyphens: manual; }}
th {{ background: {NAVY}; color: #ffffff; font-weight: bold;
     border-color: {NAVY}; }}
table tr:nth-child(even) td {{ background: {TINT}; }}

/* Tabela de identificacao no topo (sem cabecalho): 1a coluna como rotulo */
body > table:first-of-type thead {{ display: none; }}
body > table:first-of-type td:first-child,
body > table:first-of-type td:first-child strong {{ background: {NAVY};
  color: #ffffff; font-weight: bold; width: 27%; font-size: 9pt;
  letter-spacing: 0.2pt; }}
body > table:first-of-type tr:nth-child(even) td:first-child {{ background: {NAVY}; }}
"""
    extra = {
        "etapa1_modelagem": """
table.tbl-3 { table-layout: fixed; }
table.tbl-3 td:nth-child(1), table.tbl-3 th:nth-child(1) { width: 20%; }
table.tbl-3 td:nth-child(2), table.tbl-3 th:nth-child(2) { width: 27%; }
table.tbl-3 td:nth-child(3), table.tbl-3 th:nth-child(3) { width: 53%; }
""",
        "etapa2_dados": """
/* matriz de correlacao (5a tabela do documento): numeros alinhados a direita */
table.tbl-5 td, table.tbl-5 th { text-align: right; }
table.tbl-5 td:first-child, table.tbl-5 th:first-child { text-align: left; }
""",
    }.get(base, "")
    return css + extra


def build(base):
    src = RELATORIOS / f"{base}.md"
    pdf_out = RELATORIOS / f"{base}.pdf"
    text = src.read_text(encoding="utf-8")
    body = markdown.markdown(
        text, extensions=["tables", "sane_lists", "attr_list", "fenced_code"]
    )
    # numera as tabelas (tbl-1, tbl-2, ...) para permitir CSS por tabela
    contador = [0]

    def marcar(_m):
        contador[0] += 1
        return f'<table class="tbl-{contador[0]}">'

    body = re.sub(r"<table>", marcar, body)
    html = (
        "<!DOCTYPE html><html lang='pt-BR'><head><meta charset='utf-8'>"
        f"<style>{montar_css(base)}</style></head><body>{body}</body></html>"
    )
    WHTML(string=html).write_pdf(str(pdf_out))
    print(f"PDF: {pdf_out} ({pdf_out.stat().st_size} bytes)")


def main():
    bases = sys.argv[1:] or [
        "etapa1_modelagem", "etapa2_dados", "etapa3_classificador", "etapa4_resultados",
    ]
    for base in bases:
        build(base)


if __name__ == "__main__":
    main()
