#!/usr/bin/env bash
set -euo pipefail

# gather test files (exclude integration tests)
mapfile -d '' -t files < <(find test -type f -name 'test_*.py' ! -name 'test_integration*.py' -print0)

if [ "${#files[@]}" -eq 0 ]; then
  echo "No unit tests found"
  exit 1
fi

# run a single python invocation with PYTHONPATH=src and all test files
PYTHONPATH=src python -m unittest -v "${files[@]}"