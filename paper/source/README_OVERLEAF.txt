ICASSP 2027 SRVR — Overleaf upload package (v110)

This archive is intentionally flat and contains only the active manuscript.
It does not contain the nested official template that caused duplicate
Template.tex and refs.bib files in the archival release package.

Recommended import:
1. In Overleaf, choose New Project > Upload Project.
2. Upload this ZIP as a new project; do not merge it into the old project.
3. In Menu, set Main document to Template.tex.
4. Keep the compiler as pdfLaTeX.
5. Choose Recompile from scratch once.

Expected checks after compilation:
- The PDF has 5 pages.
- No citation appears as [?].
- References are numbered continuously [1] through [30].
- SVHalluc is [1], ChronusOmni is [14], CrossTask is [29], and Phi-4 is [30].

Template.tex is the main file. Do not compile paper.tex by itself.
The bibliography command intentionally reads both files:
  \\bibliography{refs,chronusomni_reference_v1}

For a local build:
  bash build.sh

Equivalent sequence:
  pdflatex Template.tex
  bibtex Template
  pdflatex Template.tex
  pdflatex Template.tex

Template.bbl is included as a resolved fallback, but both .bib files are also
present so Overleaf can regenerate the bibliography normally.
