#!/usr/bin/env bash
# Build effect-torch's native addons (Metal backend and tokenizers) and type declarations in the
# submodule, with its own pinned toolchains. Needed until effect-torch publishes its packages.
set -euo pipefail
cd "$(dirname "$0")/../vendor/effect-torch"
nix develop --command bash -c '
  pnpm install --frozen-lockfile --silent
  for package in backend-apple-native tokenizers; do
    (cd "packages/$package" && node ../../scripts/build-native.mjs --host --profile release)
  done
  # Type declarations, so Idiolect typechecks against them rather than effect-torch'"'"'s
  # sources under Idiolect'"'"'s stricter compiler settings.
  (cd packages/core && pnpm run --silent build)
  for package in backend-apple-native tokenizers; do
    (cd "packages/$package" && pnpm run --silent build:ts)
  done
'
# The build's own packages include its Effect version; remove them so effect-torch's sources
# resolve Idiolect's single copy of Effect.
rm -rf node_modules packages/*/node_modules
