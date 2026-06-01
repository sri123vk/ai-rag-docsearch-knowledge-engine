#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHONPATH="$PROJECT_ROOT/src" python3 "$PROJECT_ROOT/scripts/demo.py"

