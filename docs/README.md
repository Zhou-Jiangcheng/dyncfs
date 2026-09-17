# Documentation development

Sphinx + MyST Markdown + PyData theme, following the companion pygrnwang
documentation layout. The site is published at
https://zhou-jiangcheng.github.io/dyncfs/ by .github/workflows/docs.yml on
every push to main.

Use Python 3.12 and install docs/requirements.txt in an isolated environment:

    python -m pip install -r docs/requirements.txt
    python docs/api/check_coverage.py
    python -m sphinx -b html -W --keep-going docs docs/_build/html
    python -m http.server 8000 --bind 127.0.0.1 --directory docs/_build/html

Open http://127.0.0.1:8000/ or docs/_build/html/index.html.

The build parses source signatures and never imports DynCFS, runs solvers
or installs its runtime dependencies. The static tutorial runs separately:

    conda run -n cfs python docs/examples/quickstart.py

See development.md for the full workflow and validation.md for observed
results. Build output and tutorial output are ignored by Git through the
existing docs/_build/ rule.
