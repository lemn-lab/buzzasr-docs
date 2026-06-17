# BuzzASR docs

Documentation site for the BuzzASR project. Built with [MkDocs](https://www.mkdocs.org/) +
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/), deployed to GitHub Pages.

## Local preview

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdocs serve
# → open http://127.0.0.1:8000
```

The dev server hot-reloads on file changes.

## Build static site

```bash
mkdocs build      # outputs to site/
```

## Deploy

Push to `main` and the GitHub Actions workflow (`.github/workflows/deploy.yml`)
publishes to `gh-pages` branch. GitHub Pages serves from there.

To enable Pages on the GitHub repo:
- Settings → Pages → Source: "Deploy from a branch" → Branch: `gh-pages` / `/ (root)`

## Editing

All content lives in `docs/`. Each page is a plain markdown file. Navigation is
controlled by `mkdocs.yml`'s `nav:` block.

To add a new page:

1. Create `docs/<section>/<new-page>.md`
2. Add an entry under the appropriate section in `mkdocs.yml`'s `nav:`
3. `mkdocs serve` to preview

## License

MIT (matches the BuzzASR code release).
