#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/bump-version.sh <new-version>
# Updates version across all packages:
#   - crates/tokie/Cargo.toml
#   - crates/tokie-python/Cargo.toml
#   - crates/tokie-python/pyproject.toml

if [ $# -ne 1 ]; then
    echo "Usage: $0 <new-version>"
    echo "Example: $0 0.1.0"
    exit 1
fi

NEW_VERSION="$1"

# Validate semver format
if ! echo "$NEW_VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$'; then
    echo "Error: Version must be valid semver (e.g., 0.1.0, 1.0.0-beta.1)"
    exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Bumping all packages to v${NEW_VERSION}..."

# crates/tokie/Cargo.toml
sed -i '' "s/^version = \".*\"/version = \"${NEW_VERSION}\"/" "$ROOT/crates/tokie/Cargo.toml"
echo "  Updated crates/tokie/Cargo.toml"

# crates/tokie-python/Cargo.toml
sed -i '' "s/^version = \".*\"/version = \"${NEW_VERSION}\"/" "$ROOT/crates/tokie-python/Cargo.toml"
echo "  Updated crates/tokie-python/Cargo.toml"

# crates/tokie-python/pyproject.toml
sed -i '' "s/^version = \".*\"/version = \"${NEW_VERSION}\"/" "$ROOT/crates/tokie-python/pyproject.toml"
echo "  Updated crates/tokie-python/pyproject.toml"

# Update lockfile
cd "$ROOT"
cargo generate-lockfile 2>/dev/null
echo "  Updated Cargo.lock"

echo ""
echo "Done! All packages set to v${NEW_VERSION}"
echo ""
echo "Next steps:"
echo "  git add -A && git commit -m 'Bump version to v${NEW_VERSION}'"
echo "  git tag v${NEW_VERSION}"
echo "  gh release create v${NEW_VERSION}"
