# SRVR

Speech-Referenced Visual Re-query (SRVR) research materials.

## Current paper

The latest identified submission package is ICASSP 2027 version 110 (4 September 2026).

- Review copy: [`paper/release/SRVR_v110.pdf`](paper/release/SRVR_v110.pdf)
- Editable LaTeX source and build files: [`paper/source/`](paper/source/)
- The flat source package is the Overleaf-ready v110 upload. Set `Template.tex` as the main document and compile with pdfLaTeX.
- Research code: [`code/audio/`](code/audio/) and [`code/analysis/`](code/analysis/)
- Earlier SRVR/CMTB preparation scripts: [`code/legacy/`](code/legacy/)

## Code

`code/audio/` contains the CMTB audio-control preparation pipeline and related phases. `code/analysis/` contains the SRVR/CMTB preparation, audit, analysis, and figure scripts used across the paper experiments. `code/legacy/` preserves the earlier SRVR/CMTB scripts from the initial study phase. These scripts expect local benchmark media, model outputs, and model installations, which are not included here. Some scripts retain the original workstation paths; update their path constants for your environment before running them. The Phase 0 overview and gates are documented in [`code/audio/README.md`](code/audio/README.md).

The workspace also contains newer exploratory drafts and experiments. They are not copied here because they are not a newer released SRVR submission package. Update this repository when a later reviewed release is selected.
