# TODO

## Performance

- bench-process
  - ergonomics improvement: command line argument for number of iterations
  - more realistic benchmark? with more variation over iterations,
    include noise gen, possibly slower changes?
- avx audio gen
  - write directly into left and right output buffers? sample iteration would
    need to be in shape of [l, l, l, l, r, r, r, r] then
  - think about doing phase calculation closer to where data is used
  - interpolation for processing parameters every sample? Build long arrays
    here too?

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