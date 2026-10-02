# GitHub release guide

## Purpose

This working directory is an active research archive. A GitHub repository should contain code and reproducibility metadata, while raw data, checkpoints, Aspen files, and generated run directories remain in controlled storage or a versioned archival release.

The root `.gitignore` implements this policy. It excludes `data/`, `outputs/`, model artefacts, simulator files, local caches, Aspen validation workbooks, and the local methodology report. `data/README.md` remains tracked to document the required layout. Ignoring or untracking an artifact does not delete the local copy.

The public repository is `https://github.com/NSLab-CUK/ProcGNN`. The earlier FlowGNN repository is retained as the legacy source; it is not deleted by this migration.

## Public repository structure

```text
configs/                 Versioned experiment and model configuration
docs/                    Protocols, model documentation, and experiment catalogue
economic module GA/module/ Source code for GNN-coupled economic optimisation
scripts/                 Runnable training, evaluation, aggregation, and plotting scripts
src/process_graph/       Reusable package implementation
tests/                   Regression and smoke tests
data/README.md           Data access and local layout documentation
README.md                Installation, reproduction, and publication overview
requirements.txt         Python dependencies
```

## Keep out of Git

| Material | Location/pattern | Release route |
|---|---|---|
| Raw stream/process data and split contents | `data/` | Approved data archive or controlled access |
| Checkpoints, TensorBoard files, logs, and rendered figures | `outputs/` | DOI-backed supplementary archive when approved |
| Aspen Plus simulation files and revalidation workbook | `*.bkp`, `*.apw*`, `*.asp`, workbook exports | Controlled access unless redistribution is approved |
| Environment and editor caches | `.venv/`, `__pycache__/`, `.pytest_cache/` | Never release |

## Migration and commit identity

The 2026-10-02 migration publishes a new source snapshot on top of the destination's existing initial README commit. It does not merge the legacy FlowGNN commit history. Original history is saved in a Git bundle outside the working tree; historical author metadata is not rewritten or falsely reattributed.

For subsequent commits, this checkout uses repository-local author settings associated with the requested GitHub account:

```bash
git config --local user.name "Junhee Cho"
git config --local user.email "225366080+JunheeCho3337@users.noreply.github.com"
git remote get-url origin
git status --short
git diff --cached --check
git commit -m "Describe the reviewed change"
git push origin main
```

Stage only reviewed source/documentation changes before committing. Verify that no proprietary material or credentials have been copied into tracked paths. Do not use `git push --all` or push legacy refs/bundles to the new repository. GitHub's contributor display derives from commit history and account-linked author emails; a `.mailmap` or local username change alone does not remove already published legacy commits. The snapshot migration avoids introducing them in the first place.

## Before publishing

1. Choose an institutionally approved software licence and add `LICENSE`.
2. Add the final author list, manuscript title, and DOI through `CITATION.cff`.
3. Replace local server paths in public documentation with relative paths or archive identifiers.
4. Run the smoke tests stated in the root README in a clean environment.
5. Upload approved data, checkpoints, and final figures to a frozen archive; record its DOI, version, and checksum manifest.
6. Tag the exact source revision used for the manuscript.

## Local organisation rule

Do not move historical configurations or completed output roots solely for a GitHub cleanup: their paths are referenced by metadata and manuscript assets. For new work, use the production output pattern in `docs/REPOSITORY_LAYOUT_260812.md`:

```text
YYYYMMDD_<scope>_<model>_<purpose>/
```

Use `smoke`, `debug`, `diag`, `profile`, `perf`, or `probe` in temporary output root names to identify their purpose. All existing outputs remain protected, regardless of name.

The cleanup helper is dry-run by default. `-Execute` moves only disposable caches and reviewed obsolete source/draft paths into a sibling `_ProcGNN_*` backup. Its `restore_manifest.csv` gives original paths; `file_checksums.csv` records archived file hashes. Data, checkpoints, workbooks, reports, and generated outputs are not cleanup targets.
