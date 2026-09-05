# Engineering Rules

- Target Python 3.12 and use `uv` for dependency, environment, and command management.
- Keep the `src/knowledge_scope` layout and prefer small, explicit, typed modules over speculative abstractions.
- Put runtime dependencies in `[project.dependencies]` and development tools in the `dev` dependency group.
- Load application configuration through `knowledge_scope.shared.config.Settings`; use the `KNOWLEDGE_SCOPE_` environment-variable prefix.
- Never commit `.env` files, credentials, tokens, or other secrets. Update `.env.example` when a non-secret setting changes.
- Keep README claims aligned with implemented behavior. Do not introduce later-stage RAG infrastructure before its phase is approved.
- Add focused pytest coverage for behavior and run the Ruff checks, format check, and test suite before handing off changes.
- Preserve unrelated working-tree changes and stage or commit only explicitly requested files.
