#!/usr/bin/env bash
# Upload the release files to GitHub. Usage: scripts/publish_release.sh [target-branch]
set -euo pipefail
cd "$(dirname "$0")/.."
version=$(uv run python -c "from pedalpulse import config; print(config.DATA_VERSION)")
release_dir=$(uv run python -c "from pedalpulse import config; print(config.RELEASE_DIR)")
target=${1:-$(git branch --show-current)}
(cd "$release_dir" && sha256sum --check --quiet SHA256SUMS)
gh release create "$version" "$release_dir"/* \
  --target "$target" \
  --title "PedalPulse data $version" \
  --notes "Station by UTC-hour data for 2023-01 to 2026-03. Columns and rules: docs/data_contract.md. Check the files with: sha256sum --check SHA256SUMS"
