#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

REMOTE="${GIT_REMOTE:-origin}"
REMOTE_URL="${GIT_REMOTE_URL:-git@github.com:fffffffuuuuuccckkk/nuestg.git}"
BRANCH="${GIT_BRANCH:-$(git branch --show-current 2>/dev/null || true)}"
BRANCH="${BRANCH:-main}"
MESSAGE="${1:-Update code $(date '+%Y-%m-%d %H:%M:%S')}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: this script must be run inside a git repository" >&2
  exit 1
fi

if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  git remote add "$REMOTE" "$REMOTE_URL"
fi

git add -A -- \
  . \
  ':(exclude)datasets' \
  ':(exclude)datasets/**' \
  ':(exclude)outputs' \
  ':(exclude)outputs/**' \
  ':(exclude)checkpoints' \
  ':(exclude)checkpoints/**' \
  ':(exclude)runs' \
  ':(exclude)runs/**' \
  ':(exclude)wandb' \
  ':(exclude)wandb/**' \
  ':(exclude)logs' \
  ':(exclude)logs/**' \
  ':(exclude)**/__pycache__/**' \
  ':(exclude)**/*.pyc' \
  ':(exclude)**/*.pt' \
  ':(exclude)**/*.pth' \
  ':(exclude)**/*.ckpt' \
  ':(exclude)**/*.safetensors' \
  ':(exclude)**/*.npy' \
  ':(exclude)**/*.npz' \
  ':(exclude)**/*.h5' \
  ':(exclude)**/*.hdf5' \
  ':(exclude)**/*.pkl' \
  ':(exclude)**/*.zip' \
  ':(exclude)**/*.tar' \
  ':(exclude)**/*.tar.gz'

blocked="$(
  git diff --cached --name-only |
    grep -E '(^|/)(datasets|outputs|checkpoints|runs|wandb|logs)(/|$)|\.(pt|pth|ckpt|safetensors|npy|npz|h5|hdf5|pkl|zip|tar|tar\.gz)$' || true
)"

if [[ -n "$blocked" ]]; then
  echo "error: refusing to commit data/checkpoint/archive files:" >&2
  echo "$blocked" >&2
  echo "hint: update .gitignore or unstage these files before pushing." >&2
  exit 1
fi

if git diff --cached --quiet; then
  echo "No code changes to commit. Pushing current $BRANCH to $REMOTE..."
else
  echo "Committing these files:"
  git diff --cached --name-only | sed 's/^/  /'
  git commit -m "$MESSAGE"
fi

git push -u "$REMOTE" "$BRANCH"
