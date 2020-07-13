#!/bin/sh

cd octasine

cargo +nightly build --release --bin bench-process --features "simd with-coz"

cd ..

coz run --- target/release/bench-process