#!/usr/bin/env bash

cd build
make -j 4 #VERBOSE=1
./test