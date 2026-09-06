# Engineering Rules

- Target Python 3.12 and use `uv` for dependency, environment, and command management.
- Keep the `src/knowledge_scope` layout and prefer small, explicit, typed modules over speculative abstractions.
- Put runtime dependencies in `[project.dependencies]` and development tools in the `dev` dependency group.
- Load application configuration through `knowledge_scope.shared.config.Settings`; use the `KNOWLEDGE_SCOPE_` environment-variable prefix.
- Use Alembic as the database schema source of truth; keep sessions request-scoped and never call `create_all()` during application startup.
- Never commit `.env` files, credentials, tokens, or other secrets. Update `.env.example` when a non-secret setting changes.
- Keep README claims aligned with implemented behavior. Do not introduce later-stage RAG infrastructure before its phase is approved.
- Human-facing project documentation should default to Simplified Chinese. Agent-only instructions, source-code identifiers, commands, configuration keys, protocols, and other technical identifiers may remain in English; do not translate identifiers merely for consistency.
- In `frontend/`, use Vue 3 and TypeScript with Composition API and `<script setup>`; keep server state in Vue Query, use Pinia only for client/UI state, and do not add fake business data or unimplemented UI claims.
- Add focused pytest coverage for behavior and run the Ruff checks, format check, and test suite before handing off changes.
- Preserve unrelated working-tree changes and stage or commit only explicitly requested files.

## Frontend product UI rules

- Keep the normal UI content-first and user-facing: do not surface phase metadata, development diagnostics, raw API details, or unimplemented product areas.
- Treat the global tokens in "frontend/src/styles/main.css" as the visual source of truth; prefer restrained surfaces, clear hierarchy, and concise Simplified Chinese copy.
- Reuse the existing Element Plus components for forms, dialogs, confirmation, loading, and pagination. Keep server state in Vue Query and reserve Pinia for genuine UI state.
- Add interaction feedback and keyboard-visible focus states while respecting "prefers-reduced-motion"; avoid decorative animation and "transition: all".
