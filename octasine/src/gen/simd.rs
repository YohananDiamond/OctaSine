//! Audio generation using explicit simd
//! 
//! At least SSE2 is required. Simdeez scalar fallback will fail due to the
//! method used for stereo modulation input calculation.
//! 
//! TODO:
//!   - Interpolation for processing parameters every sample? Build long arrays here too?

use core::arch::x86_64::*;

use arrayvec::ArrayVec;
use vst::buffer::AudioBuffer;
use sleef_sys::Sleef_sind4_u35;

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
    let num_samples = audio_buffer.samples();

    let mut audio_buffer_outputs = audio_buffer.split().1;
    let audio_buffer_lefts = audio_buffer_outputs.get_mut(0);
    let audio_buffer_rights = audio_buffer_outputs.get_mut(1);

    let num_passes = num_samples / SAMPLE_PASS_SIZE;

    for pass_index in 0..num_passes {
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
        let mut voice_envelope_volumes: ArrayVec<[[[f64; SAMPLE_PASS_SIZE * 2]; 4]; 128]> = ArrayVec::new();
        let mut voice_phases: ArrayVec<[[[f64; SAMPLE_PASS_SIZE * 2]; 4]; 128]> = ArrayVec::new();
        let mut key_velocities: ArrayVec<[f64; 128]> = ArrayVec::new();
        
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
                for operator_index in 0..4 {
                    let last_phase = voice.operators[operator_index].last_phase.0;
                    let frequency = voice_base_frequency * operator_frequency_modifiers[operator_index];
                    let phase_addition = frequency * time_per_sample.0;

                    let mut new_phase = 0.0;

                    for i in 0..SAMPLE_PASS_SIZE {
                        // Do multiplication instead of successive addition for less precision loss (hopefully)
                        new_phase = last_phase + phase_addition * ((i + 1) as f64);

                        let j = i * 2;

                        operator_phases[operator_index][j] = new_phase;
                        operator_phases[operator_index][j + 1] = new_phase;
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

        let zero_value_limit_splat = _mm256_set1_pd(ZERO_VALUE_LIMIT);

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
                
                // --- White noise audio generation

                if operator_wave_type[operator_index] == WaveType::WhiteNoise {
                    let random_numbers = {
                        let mut random_numbers = [0.0f64; SAMPLE_PASS_SIZE * 2];

                        for i in 0..SAMPLE_PASS_SIZE {
                            let random = (octasine.processing.rng.f64() - 0.5) * 2.0;

                            let j = i * 2;

                            random_numbers[j] = random;
                            random_numbers[j + 1] = random;
                        }

                        random_numbers
                    };

                    let modulation_target = operator_modulation_targets[operator_index];

                    let constant_power_panning = {
                        let mut data = [0.0f64; 8];

                        let left_and_right = operators[operator_index].panning.left_and_right;
                        
                        for (i, v) in data.iter_mut().enumerate() {
                            *v = left_and_right[i % 2];
                        }

                        _mm256_loadu_pd(&data[0])
                    };

                    let operator_volume_splat = _mm256_set1_pd(operator_volume[operator_index]);
                    let operator_additive_splat = _mm256_set1_pd(operator_additive[operator_index]);

                    for i in (0..SAMPLE_PASS_SIZE * 2).step_by(4){
                        let envelope_volume = _mm256_loadu_pd(&voice_envelope_volumes[voice_index][operator_index][i]);
                        let volume_product = _mm256_mul_pd(operator_volume_splat, envelope_volume);

                        let sample = _mm256_loadu_pd(&random_numbers[i]);

                        let sample_adjusted = _mm256_mul_pd(sample, _mm256_mul_pd(volume_product, constant_power_panning));
                        let additive_out = _mm256_mul_pd(sample_adjusted, operator_additive_splat);
                        let modulation_out = _mm256_sub_pd(sample_adjusted, additive_out);

                        // Add modulation output to target operator's modulation inputs
                        let modulation_sum = _mm256_add_pd(
                            _mm256_loadu_pd(&voice_modulation_inputs[modulation_target][i]),
                            modulation_out
                        );
                        _mm256_storeu_pd(&mut voice_modulation_inputs[modulation_target][i], modulation_sum);

                        // Add additive output to summed_additive_outputs
                        let summed_plus_new = _mm256_add_pd(
                            _mm256_loadu_pd(&summed_additive_outputs[i]),
                            _mm256_mul_pd(additive_out, key_velocity_splat)
                        );
                        _mm256_storeu_pd(&mut summed_additive_outputs[i], summed_plus_new);
                    }

                    continue;
                }

                // --- Sine frequency modulation audio generation: setup operator SIMD vars

                let operator_volume_splat = _mm256_set1_pd(operator_volume[operator_index]);
                let operator_feedback_splat = _mm256_set1_pd(operator_feedback[operator_index]);
                let operator_additive_splat = _mm256_set1_pd(operator_additive[operator_index]);
                let operator_modulation_index_splat = _mm256_set1_pd(operator_modulation_index[operator_index]);

                let (pan_tendency, one_minus_pan_tendency) = {
                    // Get panning as value between -1 and 1
                    let pan_transformed = 2.0 * (operator_panning[operator_index] - 0.5);

                    let r = pan_transformed.max(0.0);
                    let l = (pan_transformed * -1.0).max(0.0);

                    // Width 8 in case of eventual avx512 support in simdeez
                    let data = [l, r, l, r, l, r, l, r];
                    
                    let tendency = _mm256_loadu_pd(&data[0]);
                    let one_minus_tendency = _mm256_sub_pd(_mm256_set1_pd(1.0), tendency);

                    (tendency, one_minus_tendency)
                };

                let constant_power_panning = {
                    let mut data = [0.0f64; 8];

                    let left_and_right = operators[operator_index].panning.left_and_right;
                    
                    for (i, v) in data.iter_mut().enumerate() {
                        *v = left_and_right[i % 2];
                    }

                    _mm256_loadu_pd(&data[0])
                };

                let modulation_target = operator_modulation_targets[operator_index];

                // --- Create samples for both channels

                let tau_splat = _mm256_set1_pd(TAU);

                for i in (0..SAMPLE_PASS_SIZE * 2).step_by(4) {
                    let envelope_volume = _mm256_loadu_pd(&voice_envelope_volumes[voice_index][operator_index][i]);
                    let volume_product = _mm256_mul_pd(operator_volume_splat, envelope_volume);

                    // Skip generation when envelope volume or operator volume is zero.
                    // Helps performance when operator envelope lengths vary a lot.
                    // Otherwise, the branching probably negatively impacts performance.
                    /*{ FIXME
                        let volume_on = _mm256_cmp_pd(volume_product, zero_value_limit_splat, _CMP_GT_OS);

                        // Higher indeces don't really matter: if previous sample has zero
                        // envelope volume, next one probably does too. Worst case scenario
                        // is that attacks are a tiny bit slower.
                        if volume_on[0].to_bits() == 0 {
                            continue;
                        }
                    }*/

                    let modulation_in_for_channel = _mm256_loadu_pd(&voice_modulation_inputs[operator_index][i]);

                    let phase = _mm256_mul_pd(
                        _mm256_loadu_pd(&voice_phases[voice_index][operator_index][i]),
                        tau_splat
                    );

                    // Weird modulation input panning
                    // Note: breaks without VF64_WIDTH >= 2 (SSE2 or newer)
                    let modulation_in_channel_sum = {
                        // Replacing with SIMD: suitable instructions in avx:
                        //   _mm256_permute_pd with imm8 = [1, 0, 1, 0] followed by addition
                        //     Indices:
                        //       0 -> 1
                        //       1 -> 0
                        //       2 -> 3
                        //       3 -> 2
                        //   _mm256_hadd_pd (takes two variables which would need to be identical): pretty slow
                        // So the idea is to take modulation_in_for_channel and run any of the above on it.

                        let mut permuted = [0.0f64; 8]; // Width 8 in case of eventual avx512 support in simdeez

                        // Should be equivalent to simd instruction permute_pd with imm8 = [1, 0, 1, 0]
                        for (j, input) in (&voice_modulation_inputs[operator_index][i..i + 4]).iter().enumerate(){
                            let add = (j + 1) % 2;
                            let subtract = j % 2;

                            permuted[j + add - subtract] = *input;
                        }

                        _mm256_add_pd(
                            _mm256_loadu_pd(&permuted[0]),
                            modulation_in_for_channel
                        )
                    };

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

                    let sample = Sleef_sind4_u35(sin_input);

                    let sample_adjusted = _mm256_mul_pd(
                        sample,
                        _mm256_mul_pd(volume_product, constant_power_panning)
                    );
                    let additive_out = _mm256_mul_pd(
                        sample_adjusted,
                        operator_additive_splat
                    );
                    let modulation_out = _mm256_sub_pd(sample_adjusted, additive_out);

                    // Add modulation output to target operator's modulation inputs
                    let modulation_sum = _mm256_add_pd(
                        _mm256_loadu_pd(&voice_modulation_inputs[modulation_target][i]),
                        modulation_out
                    );
                    _mm256_storeu_pd(&mut voice_modulation_inputs[modulation_target][i], modulation_sum);

                    // Add additive output to summed_additive_outputs
                    let summed_plus_new = _mm256_add_pd(
                        _mm256_loadu_pd(&summed_additive_outputs[i]),
                        _mm256_mul_pd(additive_out, key_velocity_splat)
                    );
                    _mm256_storeu_pd(&mut summed_additive_outputs[i], summed_plus_new);
                } // End of sample pass size *  2 iteration
            } // End of operator iteration
        } // End of voice iteration

        // --- Summed additive outputs: apply master volume and hard limit.

        let master_volume_factor_splat = _mm256_set1_pd(master_volume_factor);
        let max_volume_splat = _mm256_set1_pd(5.0);
        let min_volume_splat = _mm256_set1_pd(-5.0);

        for i in (0..SAMPLE_PASS_SIZE * 2).step_by(4) {
            let additive_outputs = _mm256_loadu_pd(&summed_additive_outputs[i]);
            let additive_outputs = _mm256_mul_pd(additive_outputs, master_volume_factor_splat);
            let limited_outputs = _mm256_max_pd(
                _mm256_min_pd(additive_outputs, max_volume_splat),
                min_volume_splat
            );

            _mm256_storeu_pd(&mut summed_additive_outputs[i], limited_outputs);
        }

        // --- Write additive outputs to audio buffer

        let sample_offset = pass_index * SAMPLE_PASS_SIZE;

        for i in 0..SAMPLE_PASS_SIZE {
            let j = i * 2;
            audio_buffer_lefts[i + sample_offset] = summed_additive_outputs[j] as f32;
            audio_buffer_rights[i + sample_offset] = summed_additive_outputs[j + 1] as f32;
        }
    } // End of pass iteration
}