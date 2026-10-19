#!/usr/bin/env bash
set -euo pipefail

# Build the ICASSP manuscript with all bibliography and cross-references.
pdflatex -interaction=nonstopmode -halt-on-error Template.tex
bibtex Template
pdflatex -interaction=nonstopmode -halt-on-error Template.tex
pdflatex -interaction=nonstopmode -halt-on-error Template.tex

echo "Built Template.pdf with BibTeX citations and LaTeX cross-references resolved."
