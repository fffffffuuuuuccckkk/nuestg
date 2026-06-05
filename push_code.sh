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

# Stage tracked modifications/deletions first, then add only untracked files
# that are not ignored by .gitignore. This avoids failing on ignored local
# artifacts such as checkpoints while still picking up new source files.
git add -u

untracked_files=()
while IFS= read -r -d '' file; do
  untracked_files+=("$file")
done < <(git ls-files -o --exclude-standard -z)

if (( ${#untracked_files[@]} > 0 )); then
  git add -- "${untracked_files[@]}"
fi

# blocked="$(
#   git diff --cached --name-only |
#     grep -E '(^|/)(datasets|outputs|checkpoints|runs|wandb|logs)(/|$)|\.(pt|pth|ckpt|safetensors|npy|npz|h5|hdf5|pkl|zip|tar|tar\.gz)$' || true
# )"
blocked="$(
  git diff --cached --name-only |
    grep -E '(^|/)(datasets|outputs|runs|wandb|logs)(/|$)|\.(pt|pth|ckpt|safetensors|onnx|bin|joblib|npy|npz|h5|hdf5|pkl|zip|tar|tar\.gz)$' || true
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
