# Documentation development

## Build locally

Use Python 3.12 for this documentation environment. This requirement is
separate from the package's declared Python 3.9 minimum.

```bash
python -m venv docs/_build/venv
```

Activate it using your shell's command, then:

```bash
python -m pip install -r docs/requirements.txt
python docs/api/check_coverage.py
python -m sphinx -b html -W --keep-going docs docs/_build/html
python -m http.server 8000 --bind 127.0.0.1 --directory docs/_build/html
```

Open `http://127.0.0.1:8000/`. You can also open
`docs/_build/html/index.html` directly; a local server is preferable for
checking browser behavior. Stop the server with Ctrl+C.

For the local Windows build created with this documentation, the isolated
environment is under `docs/_build/venv/`:

```powershell
conda run -n pygrnwang .\docs\_build\venv\Scripts\python.exe -m sphinx -b html -W --keep-going docs docs/_build/html
```

This environment was created from the activated Conda environment so the
same activation is used when invoking it. It only installs documentation
dependencies; it does not update the numerical environment.

## Build behavior

Sphinx uses MyST, sphinx-design and the PyData theme, matching the companion
pygrnwang site's visual language. The version is read from
`pyproject.toml`. API references use explicit Python-domain signatures,
verified through AST against source at every build.

The build does not import the numerical package, call setup.py, start Java
or execute example scripts. Selected tutorial figures are local static
assets. Dependencies are pinned at the direct-package level in
`docs/requirements.txt`; this is not a lock of every transitive package.

## Publication

`.github/workflows/docs.yml` runs the API check and the strict HTML build on
pull requests and on pushes to `main` that touch `docs/`, `dyncfs/`,
`pyproject.toml` or the workflow. Every run uploads the HTML as a reviewable
artifact. Pushes to `main` also deploy it to GitHub Pages at
<https://zhou-jiangcheng.github.io/dyncfs/>. The workflow can be started
manually from the Actions tab.

## Update a documented API

1. Update the explanation and exact signature on its reference page.
2. Add/remove its fully qualified name in `api/public-api.json`.
3. Update relevant configuration, unit and output descriptions.
4. Run the API check and HTML build with warnings treated as errors.
5. Run a small numerical example when the documented behavior changes.

The API check validates names and signatures, not the scientific truth of
the prose. Review source synthesis, rotation and file-writing paths together.
Keep the Chinese quickstart aligned with the executable tutorial.

## Update tutorial evidence

Run `docs/examples/quickstart.py` in an activated numerical environment
with compatible native solvers. Use a fresh output directory. Review
`summary.json`, numerical files and the rendered plot before replacing
`docs/_static/quickstart.png`. Record the environment and scope in
[validation](validation.md).

Generated libraries, virtual environments and HTML stay below the already
ignored `docs/_build/`. Existing scientific case studies remain independent
of documentation builds.
