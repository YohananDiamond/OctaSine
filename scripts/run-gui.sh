#!/bin/sh

if [[ -z $1 ]]; then
    echo "Usage: $0 [glow|wgpu|egui]"
else
    cargo run -p run_gui --features $1
fi
