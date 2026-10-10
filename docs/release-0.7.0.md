# Kimech 0.7.0 — Release acceptance

Status: **prepared, not merged or published**. Tracking PR:
[#48](https://github.com/JorgeDeLosSantos/kimech/pull/48).

## Prepared in the feature branch

- Public API and migration: `solve(..., drivers=...)`,
  `KinematicSolution.drivers`, `Configuration.drivers`,
  `SolveFailureContext.sample_index` and ordered `driver_positions`.
- `pyproject.toml` and `kimech.__version__` agree on `0.7.0`.
- `CHANGELOG.md` includes new capabilities, breaking changes, and limits.
- User guides, 2R example, five-bar and R/P numerical acceptance included.
- Visual ground supports verified with automated SVG/PNG export tests and
  manually inspected in Colab.
- GitHub Actions checks Python 3.11/3.12, package build, `twine check`,
  installation of the wheel into a clean environment, and a packaged
  Multi-DOF solve.
- Documentation uses strict Sphinx `-W` build, including Mermaid support.

## Before merge

1. Confirm all checks for the **latest PR commit** are successful.
2. Review the full PR diff and verify the intended version, links and docs.
3. Confirm the PR base is `main`, has no merge conflicts, and that the
   project owner approves integration.
4. Consider **squash merge** for a concise history; do not merge this
   preparation commit automatically.

## After approved merge

1. Check `main` HEAD and its CI; confirm package version is exactly `0.7.0`.
2. Create a release tag `v0.7.0` at the approved release commit.
3. Create/publish the GitHub release with the 0.7.0 changelog as notes.
   **Important:** `.github/workflows/publish.yml` is triggered by a published
   GitHub release and automatically attempts PyPI Trusted Publishing.
   Do **not** publish the GitHub release until PyPI credentials/environment
   are ready and publication is explicitly approved.
4. Inspect the GitHub Actions publish workflow and the resulting wheel/sdist
   on PyPI. Validate the uploaded version and metadata.
5. In a fresh environment, `python -m pip install "kimech[viz]==0.7.0"`;
   verify `kimech.__version__`, the `drivers=` API, and the 2R example.
6. Confirm the public documentation deployment, then record the release in
   the project notes/roadmap and create a clean next-development branch.

## Out of scope

No dynamics, generic geometry-based driving, global branch enumeration,
pseudo-arclength continuation or publication to TestPyPI is performed as
part of this preparation stage.
