#!/bin/sh

cd octasine

cargo +nightly run --release --bin bench-process --features "simd"
