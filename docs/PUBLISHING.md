# Publishing pi-reflex

npm package: [`pi-reflex`](https://www.npmjs.com/package/pi-reflex) · repo: `pungggi/pi-reflex`

Releases are **tag-driven via GitHub Actions** (`.github/workflows/release.yml`):
push a `v*.*.*` tag on `main` (tag MUST equal `package.json` version) → CI builds,
typechecks, tests, and publishes with OIDC trusted publishing + provenance.
Manual `npm publish` is blocked by `prepublishOnly`; `prepack` builds `dist/`
(the tarball ships `dist` + `bin`).

## Trusted Publisher setup (one-time, after first publish)

npm Trusted Publishing **cannot create a new package** — attach it only after the
first publish exists. On npmjs.com → `pi-reflex` → Settings → Trusted Publisher:

| field | value |
|---|---|
| Organization or user | `pungggi` — the **GitHub** username/org owning the repo (npm's form asks for the GitHub username, NOT the npm account; the npm package is owned by `ngsoftware`) |
| Repository | `pungggi/pi-reflex` |
| Workflow filename | `release.yml` (exact, no path) |
| Environment | *(empty)* |
| Allowed actions | tick **`npm publish`** (direct) — connections created after 2026-09-03 default to `npm stage publish` only, which stages instead of publishing |

> npm: existing trusted-publisher connections are immutable — to change any field, delete the connection and create a new one.
| Repository | `pungggi/pi-reflex` |
| Workflow filename | `release.yml` |
| Environment | *(empty)* |

## First-publish bootstrap (0.1.0 — done manually)

Because Trusted Publisher needs an existing package, 0.1.0 was published
manually from a laptop:

1. temporarily drop the `prepublishOnly` script from `package.json`
2. `npm publish --access public` **without** `--provenance` (no trusted publisher yet)
3. restore `prepublishOnly`, commit
4. configure the Trusted Publisher (table above)
5. verify the pipeline with the next release (0.1.1 via `v0.1.1` tag → OIDC + provenance)

## Release recipe (every release)

```bash
cd ~/source/pi/packages/pi-jev-jev

# 1. feature branch; tests + typecheck green before opening
git checkout -b release/x.y.z
npm test && npm run typecheck
# edit package.json version → X.Y.Z
git add -A && git commit -m "release: X.Y.Z"
git push -u origin release/x.y.z

# 2. open + merge the PR (squash keeps main linear)
gh pr create --base main --fill
gh pr merge --squash

# 3. tag the merged commit on main (tag MUST == package.json version)
git fetch origin
git checkout main && git pull --ff-only
git tag vX.Y.Z
git push origin vX.Y.Z

# 4. watch CI publish
gh run watch
npm view pi-reflex version    # expect X.Y.Z
```

## Artifacts are NOT in the npm tarball

The package is code-only (~kB). Model artifacts (~400 MB int8 / 1.6 GB fp32 per
checkpoint) resolve at runtime: `$PI_REFLEX_ARTIFACTS` → cache
(`~/.pi-reflex/engines`) → HF download from `ngSoftware/pi-reflex-artifacts`
(must be public). Generate locally with `tools/export_onnx.py` instead.
