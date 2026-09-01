# MACE--FNO report

The repository tracks the report source, bibliography, compact result macros,
and the Python source for the architecture diagram. Generate figures and
compile the document out of tree so rendered assets and TeX auxiliaries do not
accumulate in the source checkout.

For the Lucia TeX Live installation:

```bash
report_build=/path/to/external/mace_fno_report_build
figure_dir=/path/to/external/mace_fno_report_figures
tex_bin=~/texlive/2026/bin/x86_64-linux

python3 docs/report/figures/plot_fno_workflow.py \
  --output-dir "$figure_dir"

mkdir -p "$report_build"
cp docs/report/main.tex \
  docs/report/references.bib \
  docs/report/results_values.tex \
  "$figure_dir/fno_workflow.pdf" \
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
