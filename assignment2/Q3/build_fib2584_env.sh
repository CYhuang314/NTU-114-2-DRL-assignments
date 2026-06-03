#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PYBIND11_INCLUDE="${PYBIND11_INCLUDE:-/opt/pyvenv/lib/python3.13/site-packages/torch/include}"
PY_INCLUDE="$($PYTHON_BIN - <<'PY'
import sysconfig
print(sysconfig.get_paths()['include'])
PY
)"
EXT_SUFFIX="$($PYTHON_BIN - <<'PY'
import sysconfig
print(sysconfig.get_config_var('EXT_SUFFIX'))
PY
)"

c++ -O3 -Wall -shared -std=c++17 -fPIC \
  -I"$PY_INCLUDE" \
  -I"$PYBIND11_INCLUDE" \
  ./fib2584_env.cpp \
  -o ./fib2584_env${EXT_SUFFIX}

echo "Built: ./fib2584_env${EXT_SUFFIX}"
