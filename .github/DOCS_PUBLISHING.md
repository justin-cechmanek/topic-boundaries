# Publish the public docs mirror

The code repository is private. The separate
`justin-cechmanek/topic-boundaries-docs` repository is public and contains
generated Sphinx HTML/assets only. Publish after the matching code and docs
PRs merge into `main`, so the public examples never describe an unmerged API.

From the private repository root:

```bash
pip install -e ".[docs]"
sphinx-build -b html -W --keep-going docs docs/_build/public-html
git clone git@github.com:justin-cechmanek/topic-boundaries-docs.git docs/_build/public-pages
rsync -a --delete --exclude=.git --exclude=.doctrees --exclude=.buildinfo --exclude=.nojekyll docs/_build/public-html/ docs/_build/public-pages/
git -C docs/_build/public-pages status --short
```

Review that the changed paths are generated HTML, `_sources` snippets,
Sphinx assets, or the search index. Do not copy Python package files,
datasets, `TODO.txt`, or the proposal PDF. Then commit and push from
`docs/_build/public-pages`; verify the GitHub Pages build completed. The
`docs/_build` staging directories are ignored by the private repository.
