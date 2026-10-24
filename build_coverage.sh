#!/usr/bin/env sh
set -eu

cmake -S . -B build-coverage -DCMAKE_BUILD_TYPE=Debug -DNNLIB_ENABLE_COVERAGE=ON
cmake --build build-coverage --target coverage
