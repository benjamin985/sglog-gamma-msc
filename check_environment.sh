#!/usr/bin/env bash
set -u
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
if command -v matlab >/dev/null 2>&1; then
  echo "MATLAB: $(command -v matlab)"
else
  echo "MATLAB: NOT FOUND"
fi
if command -v octave >/dev/null 2>&1; then
  echo "Octave: $(command -v octave)"
else
  echo "Octave: NOT FOUND"
fi
echo "CPU cores: $(nproc)"
free -h
df -h .

