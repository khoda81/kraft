#!/usr/bin/env bash
# Requires an authenticated gh CLI with repository administration permission.
set -euo pipefail
repo=khoda81/kraft
gh repo edit "$repo" \
  --description 'Compute-aware Bayesian program mixtures in Rust, starting with finite-state predictors.' \
  --add-topic rust \
  --add-topic bayesian-inference \
  --add-topic program-synthesis \
  --add-topic finite-state-machines \
  --add-topic minimum-description-length \
  --add-topic online-learning \
  --add-topic research

gh repo view "$repo" --json nameWithOwner,description,repositoryTopics
