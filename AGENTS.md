# Agent instructions

Read and follow [CONTRIBUTING.md](CONTRIBUTING.md) before starting work. It holds
the development workflow, library contracts, verification, versioning, and
writing guidance for both humans and agents.

- Within authorized work, continue non-blocked tasks without repeated permission
  requests. Honor session authorization to push branches and open pull requests.
- Use a separate worktree when other agents are working in the repository.
- Read issue context with
  `gh issue view NUMBER --repo justin-cechmanek/topic-boundaries --json title,body,comments`.
  Inspect the private planning Project with
  `gh project item-list 1 --owner justin-cechmanek --limit 100 --format json`.
  Project access requires the GitHub CLI `project` scope. Read and update Project
  fields with `gh project` when available.
- If GitHub access is unavailable, report the limitation. Do not invent task
  state or create a second committed backlog. Temporary notes may live outside
  the repository.
- Start GitHub PR comments written by Codex with `Written by Codex (AI)`.

Adapted from the [Redis Decision Loop agent instructions](https://github.com/redis-field-engineering/redis-decision-loop/blob/main/AGENTS.md).
