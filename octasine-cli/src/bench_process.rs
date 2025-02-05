use std::collections::HashSet;
use std::time::Instant;

use colored::*;
use octasine::common::{NoteEvent, NoteEventInner};
use octasine::utils::update_audio_parameters;
use sha2::{Digest, Sha256};
use vst::plugin::PluginParameters;

use octasine::audio::gen::AudioGen;
use octasine::parameters::{OperatorParameter, Parameter, PARAMETERS};
use octasine::plugin::vst2::OctaSine;
use octasine::simd::{Simd, SimdPackedDouble};

/// Benchmark OctaSine process functions and check output sample accuracy
pub fn run() -> anyhow::Result<()> {
    // Don't forget trailing space
    let hash = "f4 7d 21 94 a6 6d f4 eb ";

    let mut all_sleef_hashes_match = true;

    let fallback_speed = {
        let (success, r) = benchmark::<octasine::simd::Fallback>("fallback", hash);

        all_sleef_hashes_match &= success;

        r
    };

    #[cfg(target_arch = "x86_64")]
    {
        let (success, r) = benchmark::<octasine::simd::Sse2>("sse2", hash);

        all_sleef_hashes_match &= success;

        println!("Speed compared to fallback:     {}x", fallback_speed / r);
    }

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx") {
        let (success, r) = benchmark::<octasine::simd::Avx>("avx", hash);

        all_sleef_hashes_match &= success;

        println!("Speed compared to fallback:     {}x", fallback_speed / r);
    }

    if all_sleef_hashes_match {
        println!(
            "\n{}",
            "All sleef output hashes matched reference hash".green()
        );

        Ok(())
    } else {
        println!(
            "\n{}",
            "Sleef output hashes didn't match reference hash".red()
        );

        Err(anyhow::anyhow!("Hashes didn't match"))
    }
}

fn benchmark<A: AudioGen + Simd>(name: &str, expected_hash: &str) -> (bool, f32) {
    const BUFFER_LEN: usize = 256;
    const BUFFER_ITERATIONS: usize = 1024 * 8;
    const NUM_VOICES: usize = 4;

    let envelope_duration_parameters: Vec<Parameter> = (0..4)
        .flat_map(|i| {
            vec![
                Parameter::Operator(i, OperatorParameter::AttackDuration),
                Parameter::Operator(i, OperatorParameter::DecayDuration),
                Parameter::Operator(i, OperatorParameter::ReleaseDuration),
            ]
        })
        .collect();

    let wave_type_parameters: Vec<Parameter> = (0..4)
        .map(|i| Parameter::Operator(i, OperatorParameter::WaveType))
        .collect();

    let parameters_to_automate: HashSet<usize> = PARAMETERS
        .iter()
        .copied()
        .filter(|p| !envelope_duration_parameters.contains(p))
        .map(|p| p.to_index() as usize)
        .collect();

    let key_on_events: Vec<NoteEvent> = (0..NUM_VOICES)
        .map(|i| NoteEvent {
            delta_frames: (i % BUFFER_LEN) as u32,
            event: NoteEventInner::Midi {
                data: [144, i as u8, 100 + i as u8],
            },
        })
        .collect();

    let key_off_events: Vec<NoteEvent> = (0..NUM_VOICES)
        .map(|i| NoteEvent {
            delta_frames: (i % BUFFER_LEN) as u32,
            event: NoteEventInner::Midi {
                data: [128, i as u8, 0],
            },
        })
        .collect();

    // Seed rng with a fixed number
    fastrand::seed(7547);

    let mut lefts = [0.0f32; BUFFER_LEN];
    let mut rights = [0.0f32; BUFFER_LEN];

    let mut octasine = OctaSine::default();
    let mut output_hasher = Sha256::new();

    for p in envelope_duration_parameters.iter() {
        match p {
            Parameter::Operator(0, _) => {
                octasine.sync.set_parameter(p.to_index() as i32, 0.5);
            }
            Parameter::Operator(1, _) => {
                octasine.sync.set_parameter(p.to_index() as i32, 0.2);
            }
            Parameter::Operator(2, _) => {
                octasine.sync.set_parameter(p.to_index() as i32, 0.1);
            }
            Parameter::Operator(3, _) => {
                octasine.sync.set_parameter(p.to_index() as i32, 0.0);
            }
            _ => unreachable!(),
        }
    }
    for p in wave_type_parameters.iter() {
        octasine.sync.set_parameter(p.to_index() as i32, 0.0);
    }

    let wave_type_parameter_indices = wave_type_parameters
        .into_iter()
        .map(|p| p.to_index() as usize)
        .collect::<Vec<_>>();

    let now = Instant::now();

    for i in 0..BUFFER_ITERATIONS {
        match i % 1024 {
            0 => {
                octasine
                    .audio
                    .enqueue_note_events(key_on_events.iter().copied());
            }
            256 => {
                octasine
                    .audio
                    .enqueue_note_events(key_off_events.iter().copied());
            }
            _ => {}
        }

        for i in 0..PARAMETERS.len() {
            // Always generate random numbers so that hash comparisons can be
            // made with/without certain parameters
            let mut value = fastrand::f32();

            if wave_type_parameter_indices.contains(&i) {
                // Avoid setting wave type to noise
                value = value * 0.79;
            }

            if parameters_to_automate.contains(&i) {
                octasine.sync.set_parameter(i as i32, value);
            }
        }

        update_audio_parameters(&mut octasine.audio, &octasine.sync);

        for (j, (lefts, rights)) in lefts
            .chunks_exact_mut(A::Pd::SAMPLES)
            .zip(rights.chunks_exact_mut(A::Pd::SAMPLES))
            .enumerate()
        {
            unsafe {
                A::process_f32(&mut octasine.audio, lefts, rights, j * A::Pd::SAMPLES);
            }
        }

        for (l, r) in lefts.iter().zip(rights.iter()) {
            output_hasher.update(l.to_ne_bytes());
            output_hasher.update(r.to_ne_bytes());
        }
    }

    let elapsed = now.elapsed();

    let elapsed_millis = elapsed.as_millis();
    let num_samples = BUFFER_LEN * BUFFER_ITERATIONS;
    let num_seconds = num_samples as f32 / 44100.0;

    let processing_time_per_sample = elapsed.as_nanos() as f32 / num_samples as f32;

    println!();
    println!(
        "--- Benchmarking OctaSine process_f32 variant: {} ---",
        name
    );
    println!("Total number of samples:        {}", num_samples);
    println!("Equivalent to audio duration:   {} seconds", num_seconds);
    println!(
        "Processing time in total:       {} milliseconds",
        elapsed_millis
    );
    println!(
        "Processing time per sample:     {} nanoseconds",
        processing_time_per_sample
    );
    println!(
        "Estimated CPU use:              {}%",
        elapsed_millis as f32 / (num_seconds * 10.0)
    );

    let output_hash: String = output_hasher
        .finalize()
        .iter()
        .take(8)
        .map(|byte| format!("{:02x} ", byte))
        .collect();

    println!("Output hash (first 8 bytes):    {}", output_hash);

    let success = output_hash == expected_hash;

    let hash_match = if success { "yes".green() } else { "no".red() };

    println!("Hash match:                     {}", hash_match);

    (success, processing_time_per_sample)
}
