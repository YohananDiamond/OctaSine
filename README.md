# OctaSine

Frequency modulation based VST2 plugin written in Rust

## About

* Four operators with independent parameters such as volume, panning,
  modulation index, feedback, three different frequency modifiers (ratio, free
  and fine) and ASDR volume envelope parameters. The operators can be
  independently switched to white noise mode
* Flexible routing allowing setting the output operator (with some
  limitations) as well as the percentage of signal that is simply added to the
  final output, enabling additive synthesis
* 128 voices (using them all simultaneously might consume quite a bit
  of CPU time though)
* Fully automatable (nice way of saying there is currently no built-in
  graphical user interface)
* Master volume and master frequency parameters

## Warning

This is not a final version of the audio rendering engine. The envelopes might
be adjusted. This means any patches you make might sound different with later
versions.

## Installation

### macOS

If you have already any of the software mentioned below, that step can be skipped.

[Install the rust compiler](https://rustup.rs/). Requires the XCode build tools from Apple, you will probably be prompted to install those.

Install nightly Rust toolchain:

```sh
rustup toolchain install nightly
```

[Install homebrew](https://brew.sh).

Install git and cmake with homebrew:

```sh
brew install git cmake
```

Clone this repository to a folder on your computer:

```sh
mkdir -p "$HOME/Downloads"
cd "$HOME/Downloads"
git clone https://github.com/greatest-ape/OctaSine.git
```

Build and install:

```sh
./scripts/macos/build-simd-and-install.sh
```

__Advanced:__ If you don't want SIMD support and/or prefer the stable toolchain, instead run:

```sh
./scripts/macos/build-and-install.sh
```

Binary (pre-built) releases might be uploaded eventually.

### Other platforms

Have a look at the cargo invocations from the macOS section scripts, they
should work fine.

## License

OctaSine is licensed under the GNU GPL 3.0. This goes for all code in this
repository not in the following list:

  * The crate simd_sleef_sin35 is licensed under the Apache 2.0 license.
  * contrib/osx_vst_bundler.sh is licensed under the MIT license. See the file
    for specifics

## Trivia

* The name OctaSine comes from the four stereo sine-wave operators
