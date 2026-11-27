# Contributing

Development takes place on the server. Use the project plan to choose a bounded change. For changes to an experimental claim, consult the team's local research protocol at `local-file/research_idea/research-protocol.md`. [Implementation plan](PLAN.md)

## Commits

- Keep each commit focused on one coherent change. Include the tests and documentation that establish the behavior in the same commit when they belong together.
- Use `type(scope): imperative summary` for the subject. Choose a concrete action and outcome, such as `fix(eval): count success on the final action` or `docs(research): repair links after moving research notes`. Common types are `feat`, `fix`, `docs`, `test`, `refactor`, `build`, and `ci`.
- Add a short body when readers need to understand a decision, a prior failure, or a change in experimental comparability. Keep the message about the work delivered in that commit.

## Pull requests and review

- Start significant changes on a short-lived branch from current `main`, such as `feat/region-decoder` or `fix/timeout-label`. Give each PR one acceptance criterion that a reviewer can check. Keep the code, tests, and docs needed to verify it together. Small independent documentation fixes can be committed directly when the maintainer requests it.
- Title the PR with the delivered behavior, using the same format as a commit subject. A single-commit PR can reuse the commit title. For several commits, name their combined outcome.
- In the body, state what changed and what now works in one to three bullets. Include the motivation only when it helps explain the diff. Under **Verification**, give commands actually run and what they established. Link a run or fixture when it makes the result reproducible.
- When the change affects experiment comparability, describe the specific change in observation access, data or split, budget, label, or metric. Record durable decisions about core interfaces, sampling, or evaluation in the relevant design document. Use a local ADR in `docs/adr/` to capture meaningful alternatives.
- Before requesting review, read the full diff and verify the acceptance criterion through the public entry point. Reviewers check the stated behavior, evidence, and research impact. The maintainer merges after review and required checks pass. Preserve shared branch history.

Use the repository template for the PR body. Track future work in the implementation plan or an issue when needed. [Pull request template](.github/pull_request_template.md) · [Implementation plan](PLAN.md)

## Engineering and experiment rules

- Use descriptive names and typed public interfaces. Keep simulator adapters independent of model implementation. Pass randomness explicitly and give collection, optimization, branching and evaluation distinct random streams.
- Every behavioral change gets a check that exercises the relevant public interface. Prioritize tests for observation leakage, terminal labels, snapshot isolation, split integrity and deterministic replay. Add checks and their real invocation to the README as implementation becomes available.
- Identify the code revision, rules and wrapper version, observation schema, experiment settings, split manifest and policy checkpoint in results. Version incompatible changes to schemas and success definitions. Preserve already reported runs and identify superseding analyses.
- Select models and thresholds on development data. Record held-out seeds before final evaluation. Store large artifacts outside Git and link to their manifests. Keep credentials and machine-specific paths in local configuration.
- Describe intended experiments in the future tense and label pilot estimates as provisional. An empirical claim names its environment, protocol, comparison and uncertainty.
