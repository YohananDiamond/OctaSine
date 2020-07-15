# TODO

- does skipping based dependency analysis change sound? and does audio sound
  ok now that parameter changes are picked up more often and parameter
  interpolation is done for each sample?
- operator_index = 3 - operator_index: this might destroy compiler
  optimizations
- dependency analysis: simplify / optimize for new code structure, also move
  to own function
- voice envelope and phase calculation could maybe be moved to own function
- think about doing phase calculation closer to where data is used??
- phase: avoid multiplication trick for "less precision", since it probably
  doesn't work any more
- premature optimization: write simd rng

## Performance

- bench-process
  - ergonomics improvement: command line argument for number of iterations
  - more realistic benchmark? with more variation over iterations,
    include noise gen, possibly slower changes?

## Other

* manual text input in parameters: DAW integration working anywhere?
* sample rate change: what needs to be done? (time reset?)
* Nice online documentation
* Consider logging when preset can't be loaded (see `load_bank_data`)

## Maybe do

* Volume shown in dB
* Iterator for presets and preset parameters
* volume off by default for operator 3 and 4. Would need to change ::default to
  ::new and this would require a refactor
* proper beta scaling - double with doubling modulator frequency
* suspend mode and so on, maybe just reset time, note time, envelopes etc on
  resume
* Use FMA again for precision, possibly enabling removing .fract() call
  in fallback sound gen?
* Fuzz Log10Table (cargo-fuzz?)
* NUM_PARAMETERS constant?
* rustfmt?

## Action on coz profiling results?

Profiling with coz (https://github.com/plasma-umass/coz) didn't result in
very relevant results. It did recommend speeding up voice generation in
general (likely envelope generation), but this was already known. It also
pointed out on line in the fallback audio gen that might be interesting. I
added a comment to it, but it's very low priority.

## Don't do

- convert_to_simd macro creates simd vars, then writes to memory, and this
  is likely counterproductive, don't do this (note: doesn't help)