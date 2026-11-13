# Contributing to Topic Boundaries

The code repository is private and retains its existing [MIT license](LICENSE).
Keep source, datasets, evaluation ratings, and selection keys private. Only
generated documentation and assets belong in the separate public docs mirror.
Follow [.github/DOCS_PUBLISHING.md](.github/DOCS_PUBLISHING.md) and publish only
after the matching changes merge and publication is authorized.

## Workflow

[README.md](README.md) describes the library and its research hypothesis.
[Boundary Methods](docs/concepts/boundary_methods.md) explains what each method
measures. Geometric boundaries alone do not establish novelty or usefulness.

- Read the assigned issue body and comments, any parent milestone and
  prerequisites, and the applicable technical contracts before changing code.
- [GitHub Issues](https://github.com/justin-cechmanek/topic-boundaries/issues) are
  the source of truth for tasks. Keep agreed scope, acceptance criteria, and
  prerequisites in issue bodies. Use comments for progress, blockers, validation
  evidence, and handoffs. Preserve existing discussion.
- The private [Project](https://github.com/users/justin-cechmanek/projects/1) holds
  delivery order and status. Use Todo for accepted work that has not started,
  In Progress while working, and Done only when acceptance criteria are met.
  Update the issue and Project together. Project status alone is not completion
  evidence. Conditional investigations remain deferred until their evidence
  prerequisites are met and the work is requested.
- Merge prerequisites before starting dependent work. Complete all contributing
  work and acceptance criteria before closing an issue or milestone.
- Use `Closes #NUMBER` only when merging satisfies every acceptance criterion.
  Otherwise use `Refs #NUMBER` and leave the issue open.
- Use one branch and reviewable PR per task. Preserve unrelated changes.
  Never use broad destructive cleanup commands. Use separate worktrees for
  concurrent work. Never merge without the user's review.
- Creating or assigning an issue does not authorize implementation. Start only
  when requested. Keep development, test data, and infrastructure local unless
  explicitly authorized. Pushing branches, opening pull requests, deployment,
  publication, and merging need authorization.
- Make the smallest complete change using existing code and tools. Avoid
  speculative frameworks, extra backends, and custom automation without a need.
- Trace callers and examples before removing code or changing public interfaces.
  Update affected callers, tests, fixtures, exports, configuration, and docs
  together. Removing an example does not prove shared code is unused.
- Keep task plans and handoffs in Issues, not a local TODO file or repository
  Markdown backlog. If GitHub is unavailable, report the limitation and keep
  temporary notes outside the repository.

## Issues, research, and writing

Use descriptive titles and explain the problem, intended result, and acceptance
criteria before implementation detail. Keep maintained code, current technical
contracts, supported examples, and reproduction instructions in git. Proposals
and responsibility planning live in Issues.

Keep one-off research in its issue. Summarize what we tested, learned, and decided
at the top, then link detailed results, environment information, and historical
experiment code. Maintain scripts or guides only when we expect to reuse them.
Put feature tests in the implementation pull request. An investigation can be
reviewed in its issue without a code merge.

Make superseded decisions clear when revising scope. Distinguish current
behavior, proposals, and untested targets. Never count skipped checks as passes.
Follow [evaluation/README.md](evaluation/README.md) for blinded human review.
Report descriptive results and limits without unsupported significance claims.

Write descriptions, issues, documentation, and review replies for someone who
understands software but has not followed this project:

- Lead with the problem, what changed, and why it matters.
- Use ordinary words. Keep implementation details only when they help the reader
  assess the change. Explain necessary terms on first use. Avoid acronyms when
  full words are clear, and preserve exact code and command names.
- Explain related work before linking issue numbers. Links supplement the
  description rather than providing the context needed to understand it.
- State what was tested, what the results support, and the main limits. Put exact
  commands, measurements, and environment details after the summary.
- Before publishing, do a separate plain-language editing pass. A reader should
  understand the purpose, result, and limits without opening another issue or
  looking up terminology. A readability score alone is not enough.

No em dashes or semicolons in docs or PR prose.

## Versions

Follow [SemVer](https://semver.org/spec/v2.0.0.html). Review public API, persisted
records and keys, configuration, and command compatibility manually. Before 1.0,
fixes bump the patch and additions or breaking changes bump the minor. Call out
breaking changes during review. Use plain `MAJOR.MINOR.PATCH` versions without
suffixes.

`topic-boundaries` is the distribution and `topic_boundaries` is its public
import. Update the version in `pyproject.toml` when a bump is required.
Documentation, tests, and development-only tooling do not require a bump unless
they change a shipped deliverable. Contributors and reviewers are responsible
for choosing and applying version bumps. No PR template or changelog is required.
Never move published tags or replace their artifacts. Tagging and publication
require separate authorization.

## Library contracts

- Examples use public imports from `topic_boundaries`. Keep optional corpus,
  visualization, and documentation dependencies in their existing extras.
- Sources return `list[Datapoint]`. Preserve document IDs and alignment between
  datapoints, vectors, and labels. Validate inputs before connecting to Redis or
  replacing an index.
- K-means and Voronoi use Euclidean geometry. The default Redis schema uses
  cosine distance. Precomputed vectors retain their supplied values. Preserve
  these distinctions and do not present method scores as one shared quality
  measure.
- `overwrite_index=True` replaces the configured Redis index and its documents.
  Use dedicated test indexes and exact cleanup. Never use `FLUSHDB`, broad cleanup
  scans, or destructive lifecycle commands against shared data.
- Schema, record, configuration, or ranking changes must update the relevant
  [architecture](docs/concepts/architecture.md), method docs, fixtures, and tests
  together. Record compatibility impact in the issue and PR.

## Verification

For code changes, run checks appropriate to the changed component and preserve
existing coverage. Tests exercise real code and services rather than mocks.
Redis-backed changes need a live Redis Stack round trip. Record actual results
and limitations in the issue or PR. The existing workflow is:

```sh
pip install -e ".[dev,corpus,viz,docs]"
docker compose up -d
python -m pytest -q
sphinx-build -b html -W --keep-going docs docs/_build/html
```

Use a dedicated local Redis instance. Test fixtures can load `.env` and use
`REDIS_URL`, so check the target before running integration tests. CI supplies
Redis Stack and optional dependencies on Python 3.10, 3.11, and 3.12. Local tests
may skip when services, models, or extras are unavailable. Report those skips.

For prose-only work, check the text and build affected documentation with warnings
as errors. Do not add tests that merely mirror a trivial edit. Package code needs
meaningful tests. Validate changed standalone scripts directly and record results.
Evaluation results must distinguish geometric behavior from human judgments of
usefulness.

