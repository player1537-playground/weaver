#!/usr/bin/env bash

root=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)

exec GraphShaderTranspiler.py \
    -i "${root:?}/foo.gsp" \
    -f element "${root:?}/graph/fs-graph-f1da6eaa-f371-40b6-81b3-34dd5f99edc1/index.bin" \
    -e GS_OUTPUT "${root:?}/foo.jpg" \
    -e GS_TILE_WIDTH 512 \
    -e GS_TILE_HEIGHT 512 \
    -e GS_TILE_Z 0 \
    -e GS_TILE_X 0 \
    -e GS_TILE_Y 0 \
    "$@" \
    ##