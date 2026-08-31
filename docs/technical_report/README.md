# Technical report

The repository tracks only the report source, bibliography, and compact
result macros. Compile it out of tree so TeX auxiliaries and the PDF do not
accumulate in the source checkout.

For the Lucia TeX Live installation used for the benchmark report:

```bash
report_build=/path/to/external/mace_fno_report_build
tex_bin=~/texlive/2026/bin/x86_64-linux

mkdir -p "$report_build"
cp docs/technical_report/main.tex \
  docs/technical_report/references.bib \
  docs/technical_report/results_values.tex \
  "$report_build/"

cd "$report_build"
"$tex_bin/pdflatex" -interaction=nonstopmode -halt-on-error main.tex
"$tex_bin/bibtex" main
"$tex_bin/pdflatex" -interaction=nonstopmode -halt-on-error main.tex
"$tex_bin/pdflatex" -interaction=nonstopmode -halt-on-error main.tex
```

`results_values.tex` is intentionally separate from the prose. Values should
be transcribed only from completed external logs and audits after validation
selection; held-out test metrics and the Au2--MgO wetting endpoints must not
be used to choose a model.

The generated `main.pdf`, TeX auxiliary files, convergence logs, checkpoints,
and audit outputs remain outside Git. Before publication, archive the exact
external evidence together with checksums and software/environment metadata.
