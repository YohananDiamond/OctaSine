//! Audio generation using explicit simd
//! 
//! At least SSE2 is required. Simdeez scalar fallback will fail due to the
//! method used for stereo modulation input calculation.
//! 
//! TODO:
//!   - Interpolation for processing parameters every sample? Build long arrays here too?

use core::arch::x86_64::*;

use arrayvec::ArrayVec;
use fastrand::Rng;
use vst::buffer::AudioBuffer;
use sleef_sys::Sleef_sind4_u35;
use itertools::izip;

use vst2_helpers::processing_parameters::ProcessingParameter;

use crate::OctaSine;
use crate::common::*;
use crate::constants::*;
use crate::processing_parameters::*;

/// Each SAMPLE_PASS_SIZE samples, load parameter changes and processing
/// parameter values (interpolated values where applicable)
const SAMPLE_PASS_SIZE: usize = 16;


#[target_feature(enable = "avx")]
pub unsafe fn process_f32_avx(
    octasine: &mut OctaSine,
    audio_buffer: &mut AudioBuffer<f32>
){
    // Per-pass voice data. Indexing: 128 voices, 4 operators
    let mut voice_envelope_volumes: ArrayVec<[[[f64; SAMPLE_PASS_SIZE * 2]; 4]; 128]> = ArrayVec::new();
    let mut voice_phases: ArrayVec<[[[f64; SAMPLE_PASS_SIZE * 2]; 4]; 128]> = ArrayVec::new();
    let mut key_velocities: ArrayVec<[f64; 128]> = ArrayVec::new();

    let mut audio_buffer_outputs = audio_buffer.split().1;
    let audio_buffer_lefts = audio_buffer_outputs.get_mut(0);
    let audio_buffer_rights = audio_buffer_outputs.get_mut(1);

    let audio_buffer_chunks = izip!(
        audio_buffer_lefts.chunks_exact_mut(SAMPLE_PASS_SIZE),
        audio_buffer_rights.chunks_exact_mut(SAMPLE_PASS_SIZE)
    );

    for (audio_buffer_left_chunk, audio_buffer_right_chunk) in audio_buffer_chunks {
        // --- Update processing parameters from preset parameters

        let changed_preset_parameters = octasine.sync_only.presets
            .get_changed_parameters();

        if let Some(indeces) = changed_preset_parameters {
            for (index, opt_new_value) in indeces.iter().enumerate(){
                if let Some(new_value) = opt_new_value {
                    if let Some(p) = octasine.processing.parameters.get(index){
                        p.set_from_preset_value(*new_value);
                    }
                }
            }
        }

        // --- Set some generally useful variables

        let operators = &mut octasine.processing.parameters.operators;

        let time_per_sample = octasine.processing.time_per_sample;
        let time = octasine.processing.global_time;
        let master_volume_factor = VOICE_VOLUME_FACTOR * octasine.processing.parameters.master_volume.get_value(time);

        // --- Get operator-only data which will be valid for whole pass and all voices.

        // Interpolated
        let mut operator_volume: [f64; 4] = [0.0; 4];
        let mut operator_modulation_index = [0.0f64; 4];
        let mut operator_feedback: [f64; 4] = [0.0; 4];
        let mut operator_panning: [f64; 4] = [0.0; 4];
        let mut operator_additive: [f64; 4] = [0.0; 4];
        
        // Not interpolated
        let mut operator_wave_type = [WaveType::Sine; 4];
        let mut operator_frequency_modifiers: [f64; 4] = [0.0; 4]; 
        let mut operator_modulation_targets = [0usize; 4];

        for (index, operator) in operators.iter_mut().enumerate(){
            operator_volume[index] = operator.volume.get_value(time);
            operator_modulation_index[index] = operator.modulation_index.get_value(time);
            operator_feedback[index] = operator.feedback.get_value(time);
            operator_panning[index] = operator.panning.get_value(time);

            // Get additive factor; use 1.0 for operator 1
            operator_additive[index] = if index == 0 {
                1.0
            } else {
                operator.additive_factor.get_value(time)
            };

            operator_wave_type[index] = operator.wave_type.value;

            operator_frequency_modifiers[index] = operator.frequency_ratio.value *
                operator.frequency_free.value * operator.frequency_fine.value;

            if let Some(p) = &mut operator.output_operator {
                use ProcessingParameterOperatorModulationTarget::*;

                let opt_value = match p {
                    OperatorIndex2(p) => Some(p.get_value(())),
                    OperatorIndex3(p) => Some(p.get_value(())),
                };

                if let Some(value) = opt_value {
                    operator_modulation_targets[index] = value;
                }
            }
        }

        // Operator dependency analysis to allow skipping audio generation when possible
        let operator_generate_audio: [bool; 4] = {
            let mut operator_generate_audio = [true; 4];
            let mut operator_additive_zero = [false; 4];
            let mut operator_modulation_index_zero = [false; 4];
            
            for operator_index in 0..4 {
                // If volume is off, just set to skippable, don't even bother with lt calculations
                if operator_volume[operator_index].lt(&ZERO_VALUE_LIMIT){
                    operator_generate_audio[operator_index] = false;
                } else {
                    operator_additive_zero[operator_index] =
                        operator_additive[operator_index].lt(&ZERO_VALUE_LIMIT);

                    operator_modulation_index_zero[operator_index] =
                        operator_modulation_index[operator_index].lt(&ZERO_VALUE_LIMIT);
                }
            }

            for _ in 0..3 {
                for operator_index in 1..4 {
                    let modulation_target = operator_modulation_targets[operator_index];

                    // Skip generation if operator was previously determined to be skippable OR
                    let skip_condition = !operator_generate_audio[operator_index] || (
                        // Additive factor for this operator is off AND
                        operator_additive_zero[operator_index] && (
                            // Modulation target was previously determined to be skippable OR
                            !operator_generate_audio[modulation_target] ||
                            // Modulation target is white noise OR
                            operator_wave_type[modulation_target] == WaveType::WhiteNoise ||
                            // Modulation target doesn't do anything with its input modulation
                            operator_modulation_index_zero[modulation_target]
                        )
                    );

                    if skip_condition {
                        operator_generate_audio[operator_index] = false;
                    }
                }
            }

            operator_generate_audio
        };

        // Necessary for interpolation
        octasine.processing.global_time.0 += time_per_sample.0 * (SAMPLE_PASS_SIZE as f64);

        // --- Collect voice data (envelope volume, phases) necessary for sound generation

        // FIXME: optimize section, possibly with simd. Maybe envelopes can be calculated less often

        // Maybe operator indexes should be inversed (3 - operator_index)
        // because that is how they will be accessed later.
        
        let mut num_active_voices = 0;

        for voice in octasine.processing.voices.iter_mut(){
            if voice.active {
                let mut operator_envelope_volumes = [[0.0f64; SAMPLE_PASS_SIZE * 2]; 4];
                let mut operator_phases = [[0.0f64; SAMPLE_PASS_SIZE * 2]; 4];

                let voice_base_frequency = voice.midi_pitch.get_frequency(
                    octasine.processing.parameters.master_frequency.value
                );

                // Envelope
                for i in 0..SAMPLE_PASS_SIZE {
                    for (operator_index, operator) in operators.iter_mut().enumerate(){
                        let v = voice.operators[operator_index].volume_envelope.get_volume(
                            &octasine.processing.log10_table,
                            &operator.volume_envelope,
                            voice.key_pressed,
                            voice.duration
                        );

                        let j = i * 2;

                        operator_envelope_volumes[operator_index][j] = v;
                        operator_envelope_volumes[operator_index][j + 1] = v;
                    }

                    voice.duration.0 += time_per_sample.0;
                }

                // Phase
                for (operator_index, phases) in operator_phases.iter_mut()
                    .enumerate()
                {
                    let last_phase = voice.operators[operator_index].last_phase.0;
                    let frequency = voice_base_frequency *
                        operator_frequency_modifiers[operator_index];
                    let phase_addition = frequency * time_per_sample.0;

                    let mut new_phase = 0.0;

                    for (i, phase_chunk) in phases.chunks_exact_mut(2)
                        .enumerate()
                    {
                        // Do multiplication instead of successive addition
                        // for less precision loss (hopefully)
                        new_phase = last_phase +
                            phase_addition * ((i + 1) as f64);

                        phase_chunk[0] = new_phase;
                        phase_chunk[1] = new_phase;
                    }

                    // Save phase
                    voice.operators[operator_index].last_phase.0 = new_phase;
                }

                voice_envelope_volumes.push(operator_envelope_volumes);
                voice_phases.push(operator_phases);
                key_velocities.push(voice.key_velocity.0);

                voice.deactivate_if_envelopes_ended();

                num_active_voices += 1;
            }
        }

        // --- Generate samples for all operators and voices

        // Sample pass size * 2 because of two channels. Even index = left channel
        let mut summed_additive_outputs = [0.0f64; SAMPLE_PASS_SIZE * 2];
        // Dummy modulation output for operator 0
        let mut dummy_modulation_out = [0.0f64; SAMPLE_PASS_SIZE * 2];

        // FIXME: this was previously used for skipping samples if
        // volume is off, might be useful to put back
        // let zero_value_limit_splat = _mm256_set1_pd(ZERO_VALUE_LIMIT);

        // Voice index here is not the same as in processing storage
        for voice_index in 0..num_active_voices {
            // Voice modulation input storage, indexed by operator
            let mut voice_modulation_inputs = [[0.0f64; SAMPLE_PASS_SIZE * 2]; 4];

            let key_velocity_splat = _mm256_set1_pd(key_velocities[voice_index]);

            // Go through operators downwards, starting with operator 4
            for operator_index in 0..4 { // FIXME: better iterator with 3, 2, 1, 0 possible?
                let operator_index = 3 - operator_index;

                // Possibly skip generation based on previous dependency analysis
                if !operator_generate_audio[operator_index]{
                    continue;
                }

                gen_samples_for_voice_operator(
                    &octasine.processing.rng,

                    &mut summed_additive_outputs,
                    &mut voice_modulation_inputs,
                    &mut dummy_modulation_out[..],

                    &voice_envelope_volumes[voice_index][operator_index],
                    &voice_phases[voice_index][operator_index],
                    key_velocity_splat,

                    operator_index,
                    operator_wave_type[operator_index],
                    operator_volume[operator_index],
                    operator_additive[operator_index],
                    operator_panning[operator_index],
                    operators[operator_index].panning.left_and_right,
                    operator_modulation_targets[operator_index],
                    operator_feedback[operator_index],
                    operator_modulation_index[operator_index],
                );
            } // End of operator iteration
        } // End of voice iteration

        // --- Summed additive outputs: apply master volume and hard limit.

        let master_volume_factor_splat = _mm256_set1_pd(master_volume_factor);
        let max_volume_splat = _mm256_set1_pd(5.0);
        let min_volume_splat = _mm256_set1_pd(-5.0);

        for chunk in summed_additive_outputs.chunks_exact_mut(4){
            let mut outputs = _mm256_loadu_pd(&chunk[0]);

            outputs = _mm256_mul_pd(master_volume_factor_splat, outputs);

            // Hard limit
            outputs = _mm256_min_pd(max_volume_splat, outputs);
            outputs = _mm256_max_pd(min_volume_splat, outputs);

            _mm256_storeu_pd(&mut chunk[0], outputs);
        }

        // --- Write additive outputs to audio buffer
        
        for (additive_chunk, buffer_left, buffer_right) in izip!(
            summed_additive_outputs.chunks_exact(2),
            audio_buffer_left_chunk.iter_mut(),
            audio_buffer_right_chunk.iter_mut()
        ){
            *buffer_left = additive_chunk[0] as f32;
            *buffer_right = additive_chunk[1] as f32;
        }

        // --- Clean up voice data for next pass

        voice_envelope_volumes.clear();
        voice_phases.clear();
        key_velocities.clear();
    } // End of pass iteration
}


#[inline]
#[target_feature(enable = "avx")]
unsafe fn gen_samples_for_voice_operator(
    rng: &Rng,

    additive_outputs: &mut [f64],
    voice_modulation_inputs: &mut [[f64; SAMPLE_PASS_SIZE * 2]; 4],
    dummy_modulation_out: &mut [f64],

    voice_envelope_volumes: &[f64],
    voice_phases: &[f64],
    key_velocity_splat: __m256d,

    operator_index: usize,
    operator_wave_type: WaveType,
    operator_volume: f64,
    operator_additive: f64,
    operator_panning: f64,
    operator_left_and_right: [f64; 2],
    operator_modulation_target: usize,
    operator_feedback: f64,
    operator_modulation_index: f64,
){
    let tau_splat = _mm256_set1_pd(TAU);
    let one_splat = _mm256_set1_pd(1.0);
    let two_splat = _mm256_set1_pd(2.0);

    let operator_is_noise = operator_wave_type == WaveType::WhiteNoise;
    let operator_volume_splat = _mm256_set1_pd(operator_volume);
    let operator_additive_splat = _mm256_set1_pd(operator_additive);

    let constant_power_panning = constant_power_panning_from_left_and_right(
        operator_left_and_right
    );

    let summed_additive_output_chunks = additive_outputs.chunks_exact_mut(4);
    let envelope_volume_chunks = voice_envelope_volumes.chunks_exact(4);

    if operator_is_noise {
        let modulation_out_chunks = voice_modulation_inputs[operator_modulation_target].chunks_exact_mut(4);

        let chunks = izip!(
            summed_additive_output_chunks,
            modulation_out_chunks,
            envelope_volume_chunks,
        );

        for (
            mut additive_out_chunk,
            mut modulation_out_chunk,
            envelope_volume_chunk,
        ) in chunks {
            let envelope_volume = _mm256_loadu_pd(&envelope_volume_chunk[0]);
            let volume_product = _mm256_mul_pd(operator_volume_splat, envelope_volume);

            let samples = gen_noise_samples(
                rng,
                one_splat,
                two_splat
            );

            write_samples_to_chunks(
                volume_product,
                constant_power_panning,
                operator_additive_splat,
                key_velocity_splat,
                &mut modulation_out_chunk,
                &mut additive_out_chunk,
                samples,
            );
        }
    } else {
        let operator_feedback_splat = _mm256_set1_pd(operator_feedback);
        let operator_modulation_index_splat = _mm256_set1_pd(operator_modulation_index);
        let pan_tendency = calculate_pan_tendency(operator_panning);
        let one_minus_pan_tendency = _mm256_sub_pd(one_splat, pan_tendency);

        let voice_phase_chunks = voice_phases.chunks_exact(4);

        let (
            voice_modulation_inputs_below,
            voice_modulation_inputs_above
        ) = voice_modulation_inputs.split_at_mut(operator_index);

        let modulation_in_chunks = voice_modulation_inputs_above[0].chunks_exact(4);
        let modulation_out_chunks = if operator_index == 0 {
            dummy_modulation_out.chunks_exact_mut(4)
        } else {
            voice_modulation_inputs_below[operator_modulation_target].chunks_exact_mut(4)
        };

        let chunks = izip!(
            summed_additive_output_chunks,
            modulation_out_chunks,
            modulation_in_chunks,
            envelope_volume_chunks,
            voice_phase_chunks,
        );

        for (
            mut additive_out_chunk,
            mut modulation_out_chunk,
            modulation_in_chunk,
            envelope_volume_chunk,
            voice_phase_chunk
        ) in chunks {
            let envelope_volume = _mm256_loadu_pd(&envelope_volume_chunk[0]);
            let volume_product = _mm256_mul_pd(operator_volume_splat, envelope_volume);
            let voice_phases = _mm256_loadu_pd(&voice_phase_chunk[0]);
            let modulation_in_for_channel = _mm256_loadu_pd(&modulation_in_chunk[0]);

            let samples = gen_sin_samples(
                tau_splat,
                voice_phases,
                modulation_in_for_channel,
                pan_tendency,
                one_minus_pan_tendency,
                operator_feedback_splat,
                operator_modulation_index_splat,
            );

            write_samples_to_chunks(
                volume_product,
                constant_power_panning,
                operator_additive_splat,
                key_velocity_splat,
                &mut modulation_out_chunk,
                &mut additive_out_chunk,
                samples,
            );
        }
    } // End of sample pass size *  2 iteration
}


#[inline]
#[target_feature(enable = "avx")]
unsafe fn constant_power_panning_from_left_and_right(
    operator_left_and_right: [f64; 2]
) -> __m256d {
    let [left, right] = operator_left_and_right;

    _mm256_set_pd(right, left, right, left)
}


#[inline]
#[target_feature(enable = "avx")]
unsafe fn calculate_pan_tendency(
    operator_panning: f64
) -> __m256d {
    // Get panning as value between -1 and 1
    let pan_transformed = 2.0 * (operator_panning - 0.5);

    let right = pan_transformed.max(0.0);
    let left = (pan_transformed * -1.0).max(0.0);

    _mm256_set_pd(right, left, right, left)
}


#[inline]
#[target_feature(enable = "avx")]
unsafe fn gen_sin_samples(
    tau_splat: __m256d,
    voice_phases: __m256d,
    modulation_in_for_channel: __m256d,
    pan_tendency: __m256d,
    one_minus_pan_tendency: __m256d,
    operator_feedback_splat: __m256d,
    operator_modulation_index_splat: __m256d,
) -> __m256d {
    let phase = _mm256_mul_pd(voice_phases, tau_splat);

    // Weird modulation input panning
    let modulation_in_channel_sum = _mm256_hadd_pd(modulation_in_for_channel, modulation_in_for_channel);
    let modulation_in = _mm256_add_pd(
        _mm256_mul_pd(pan_tendency, modulation_in_channel_sum),
        _mm256_mul_pd(one_minus_pan_tendency, modulation_in_for_channel)
    );

    let feedback = _mm256_mul_pd(
        operator_feedback_splat,
        Sleef_sind4_u35(phase)
    );

    let sin_input = _mm256_add_pd(
        _mm256_mul_pd(
            operator_modulation_index_splat,
            _mm256_add_pd(feedback, modulation_in)
        ),
        phase
    );

    Sleef_sind4_u35(sin_input)
}


#[inline]
#[target_feature(enable = "avx")]
unsafe fn gen_noise_samples(
    rng: &Rng,
    one_splat: __m256d,
    two_splat: __m256d,
) -> __m256d {
    let random_1 = rng.f64();
    let random_2 = rng.f64();

    let mut samples = _mm256_set_pd(
        random_1,
        random_1,
        random_2,
        random_2
    );

    samples = _mm256_sub_pd(samples, one_splat);
    samples = _mm256_mul_pd(samples, two_splat);

    samples
}


#[inline]
#[target_feature(enable = "avx")]
unsafe fn write_samples_to_chunks(
    volume_product: __m256d,
    constant_power_panning: __m256d,
    operator_additive_splat: __m256d,
    key_velocity_splat: __m256d,
    modulation_out_chunk: &mut [f64],
    additive_out_chunk: &mut [f64],
    sample: __m256d,
){
    let sample_adjusted = _mm256_mul_pd(sample, _mm256_mul_pd(volume_product, constant_power_panning));
    let new_additive_out = _mm256_mul_pd(sample_adjusted, operator_additive_splat);
    let new_modulation_out = _mm256_sub_pd(sample_adjusted, new_additive_out);

    let modulation_out_sum = _mm256_add_pd(
        _mm256_loadu_pd(&modulation_out_chunk[0]),
        new_modulation_out
    );
    _mm256_storeu_pd(&mut modulation_out_chunk[0], modulation_out_sum);

    let additive_out_sum = _mm256_add_pd(
        _mm256_loadu_pd(&additive_out_chunk[0]),
        _mm256_mul_pd(new_additive_out, key_velocity_splat)
    );
    _mm256_storeu_pd(&mut additive_out_chunk[0], additive_out_sum);
}