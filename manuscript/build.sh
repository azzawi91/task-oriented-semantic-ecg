#!/usr/bin/env bash
# Builds (1) the two-column IEEEtran PDF (page-count check, final format) and
# (2) the single-column, double-spaced, 12-pt review version required by TGCN.
set -e; cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null
cp main.pdf manuscript_twocolumn.pdf
sed -e 's/\\documentclass\[journal\]{IEEEtran}/\\documentclass[12pt,onecolumn,draftclsnofoot]{IEEEtran}\\usepackage{setspace}\\doublespacing/' \
    -e 's/\\begin{figure\*}/\\begin{figure}/g' -e 's/\\end{figure\*}/\\end{figure}/g' \
    -e 's/\\begin{table\*}/\\begin{table}/g' -e 's/\\end{table\*}/\\end{table}/g' \
    -e 's/width=0.9\\textwidth/width=0.85\\textwidth/g' -e 's/width=0.95\\textwidth/width=0.9\\textwidth/g' \
    -e 's/\\resizebox{\\columnwidth}/\\resizebox{0.9\\textwidth}/g' -e 's/width=\\columnwidth/width=0.7\\textwidth/g' main.tex > main_review.tex
pdflatex -interaction=nonstopmode main_review.tex >/dev/null && pdflatex -interaction=nonstopmode main_review.tex >/dev/null
cp main_review.pdf manuscript_review_format.pdf
grep "Output written" main.log main_review.log
