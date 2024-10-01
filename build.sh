#!/bin/bash

mkdir -p build
cd build
libtorch_path=$(pwd)/../libtorch
# absolute path to libtorch
libtorch_path=$(realpath ../libtorch)
echo "libtorch_path: $libtorch_path"
cmake -DCMAKE_PREFIX_PATH=$libtorch_path ..
cmake --build . --config Release