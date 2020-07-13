#!/bin/sh

cd octasine_vst

cargo +nightly asm "octasine::gen::simd::process_f32_avx" --rust --features "simd logging"