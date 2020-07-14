//! Audio generation using simd intrinsics (avx)
//!
//! Requires nightly because sleef-sys requires simd ffi

use core::arch::x86_64::*;

use arrayvec::ArrayVec;
use fastrand::Rng;
use itertools::izip;
use sleef_sys::Sleef_sind4_u35;
use vst::buffer::AudioBuffer;

use vst2_helpers::processing_parameters::ProcessingParameter;

use crate::OctaSine;
use crate::common::*;
use crate::constants::*;
use crate::processing_parameters::*;


// /// Each SAMPLE_PASS_SIZE samples, load parameter changes and processing
// /// parameter values (interpolated values where applicable)
// const SAMPLE_PASS_SIZE: usize = 32;


const VECTOR_WIDTH: usize = 4;


macro_rules! convert_to_simd {
    ($name:ident) => {
        let $name = [
            _mm256_loadu_pd(&$name[0][0]),
            _mm256_loadu_pd(&$name[1][0]),
            _mm256_loadu_pd(&$name[2][0]),
            _mm256_loadu_pd(&$name[3][0]),
        ];
    };
}



#[target_feature(enable = "avx")]
pub unsafe fn process_f32_avx(
    octasine: &mut OctaSine,
    audio_buffer: &mut AudioBuffer<f32>
){
    // Per-pass voice data. Indexing: 128 voices, 4 operators
    let mut voice_envelope_volumes: ArrayVec<[[__m256d; 4]; 128]> = ArrayVec::new();
    let mut voice_phases: ArrayVec<[[__m256d; 4]; 128]> = ArrayVec::new();
    let mut key_velocities: ArrayVec<[f64; 128]> = ArrayVec::new();

    let mut audio_buffer_outputs = audio_buffer.split().1;
    let audio_buffer_lefts = audio_buffer_outputs.get_mut(0);
    let audio_buffer_rights = audio_buffer_outputs.get_mut(1);

    let audio_buffer_chunks = izip!(
        audio_buffer_lefts.chunks_exact_mut(VECTOR_WIDTH),
        audio_buffer_rights.chunks_exact_mut(VECTOR_WIDTH)
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
        
        // -- Fetch non-interpolated values

        let mut operator_wave_type = [WaveType::Sine; 4];
        let mut operator_frequency_modifiers = [0.0f64; 4]; 
        let mut operator_modulation_targets = [0usize; 4];

        for (operator_index, operator) in operators.iter_mut().enumerate(){
            operator_wave_type[operator_index] = operator.wave_type.value;

            operator_frequency_modifiers[operator_index] = operator.frequency_ratio.value *
                operator.frequency_free.value * operator.frequency_fine.value;

            if let Some(p) = &mut operator.output_operator {
                use ProcessingParameterOperatorModulationTarget::*;

                let opt_value = match p {
                    OperatorIndex2(p) => Some(p.get_value(())),
                    OperatorIndex3(p) => Some(p.get_value(())),
                };

                if let Some(value) = opt_value {
                    operator_modulation_targets[operator_index] = value;
                }
            }
        }

        // -- Fetch interpolated values

        let mut master_volume_factor = [0.0f64; VECTOR_WIDTH];
        let mut operator_volume = [[0.0f64; VECTOR_WIDTH]; 4];
        let mut operator_modulation_index = [[0.0f64; VECTOR_WIDTH]; 4];
        let mut operator_feedback = [[0.0f64; VECTOR_WIDTH]; 4];
        let mut operator_panning = [[0.0f64; VECTOR_WIDTH]; 4];
        let mut operator_additive = [[0.0f64; VECTOR_WIDTH]; 4];
        let mut operator_constant_power_panning_left = [[0.0f64; VECTOR_WIDTH]; 4];
        let mut operator_constant_power_panning_right = [[0.0f64; VECTOR_WIDTH]; 4];

        let mut time = octasine.processing.global_time;

        for i in 0..VECTOR_WIDTH {
            master_volume_factor[i] = VOICE_VOLUME_FACTOR *
                octasine.processing.parameters.master_volume.get_value(time);

            for (operator_index, operator) in operators.iter_mut().enumerate(){
                operator_volume[operator_index][i] = operator.volume.get_value(time);
                operator_modulation_index[operator_index][i] = operator.modulation_index.get_value(time);
                operator_feedback[operator_index][i] = operator.feedback.get_value(time);
                operator_panning[operator_index][i] = operator.panning.get_value(time);

                operator_constant_power_panning_left[operator_index][i] =
                    operator.panning.left_and_right[0];
                operator_constant_power_panning_right[operator_index][i] =
                    operator.panning.left_and_right[1];

                // Get additive factor; use 1.0 for operator 1
                operator_additive[operator_index][i] = if operator_index == 0 {
                    1.0
                } else {
                    operator.additive_factor.get_value(time)
                };
            }

            time.0 += time_per_sample.0;
        }

        octasine.processing.global_time = time;

        // --- Operator dependency analysis to allow skipping audio generation when possible

        let mut operator_generate_audio = [[true; 4]; VECTOR_WIDTH];

        {
            let mut operator_additive_zero = [[false; 4]; VECTOR_WIDTH];
            let mut operator_modulation_index_zero = [[false; 4]; VECTOR_WIDTH];

            for i in 0..VECTOR_WIDTH {
                for operator_index in 0..4 {
                    // If volume is off, just set to skippable, don't even bother with lt calculations
                    if operator_volume[operator_index][i].lt(&ZERO_VALUE_LIMIT){
                        operator_generate_audio[i][operator_index] = false;
                    } else {
                        operator_additive_zero[i][operator_index] =
                            operator_additive[i][operator_index].lt(&ZERO_VALUE_LIMIT);

                        operator_modulation_index_zero[i][operator_index] =
                            operator_modulation_index[operator_index][i].lt(&ZERO_VALUE_LIMIT);
                    }
                }

                for _ in 0..3 {
                    for operator_index in 1..4 {
                        let modulation_target = operator_modulation_targets[operator_index];

                        // Skip generation if operator was previously determined to be skippable OR
                        let skip_condition = !operator_generate_audio[i][operator_index] || (
                            // Additive factor for this operator is off AND
                            operator_additive_zero[i][operator_index] && (
                                // Modulation target was previously determined to be skippable OR
                                !operator_generate_audio[i][modulation_target] ||
                                // Modulation target is white noise OR
                                operator_wave_type[modulation_target] == WaveType::WhiteNoise ||
                                // Modulation target doesn't do anything with its input modulation
                                operator_modulation_index_zero[i][modulation_target]
                            )
                        );

                        if skip_condition {
                            operator_generate_audio[i][operator_index] = false;
                        }
                    }
                }
            }
        }

        // --- Collect voice data (envelope volume, phases) necessary for sound generation

        // FIXME: optimize section, possibly with simd. Maybe envelopes can be calculated less often

        // Maybe operator indexes should be inversed (3 - operator_index)
        // because that is how they will be accessed later.
        
        let mut num_active_voices = 0;

        for voice in octasine.processing.voices.iter_mut(){
            if voice.active {
                let mut operator_envelope_volumes = [[0.0f64; VECTOR_WIDTH]; 4];
                let mut operator_phases = [[0.0f64; VECTOR_WIDTH]; 4];

                let voice_base_frequency = voice.midi_pitch.get_frequency(
                    octasine.processing.parameters.master_frequency.value
                );

                // Envelope
                for i in 0..VECTOR_WIDTH {
                    for (operator_index, operator) in operators.iter_mut().enumerate(){
                        let v = voice.operators[operator_index].volume_envelope.get_volume(
                            &octasine.processing.log10_table,
                            &operator.volume_envelope,
                            voice.key_pressed,
                            voice.duration
                        );

                        operator_envelope_volumes[operator_index][i] = v;
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

                    for (i, p) in phases.iter_mut().enumerate(){
                        // Do multiplication instead of successive addition
                        // for less precision loss (hopefully)
                        new_phase = last_phase +
                            phase_addition * ((i + 1) as f64);

                        *p = new_phase;
                    }

                    // Save phase
                    voice.operators[operator_index].last_phase.0 = new_phase;
                }

                convert_to_simd!(operator_envelope_volumes);
                voice_envelope_volumes.push(operator_envelope_volumes);

                convert_to_simd!(operator_phases);
                voice_phases.push(operator_phases);

                key_velocities.push(voice.key_velocity.0);

                voice.deactivate_if_envelopes_ended();

                num_active_voices += 1;
            }
        }

        // --- Generate samples for all operators and voices

        let master_volume_factor = _mm256_loadu_pd(&master_volume_factor[0]);

        convert_to_simd!(operator_volume);
        convert_to_simd!(operator_modulation_index);
        convert_to_simd!(operator_feedback);
        convert_to_simd!(operator_panning);
        convert_to_simd!(operator_additive);
        convert_to_simd!(operator_constant_power_panning_left);
        convert_to_simd!(operator_constant_power_panning_right);

        let mut additive_outputs_left = _mm256_setzero_pd();
        let mut additive_outputs_right = _mm256_setzero_pd();

        // Dummy modulation output for operator 0
        let mut dummy_modulation_out = _mm256_setzero_pd();

        let zero_splat = _mm256_setzero_pd();
        let two_splat = _mm256_set1_pd(2.0);
        let zero_point_five_splat = _mm256_set1_pd(0.5);

        // FIXME: this was previously used for skipping samples if
        // volume is off, might be useful to put back
        // let zero_value_limit_splat = _mm256_set1_pd(ZERO_VALUE_LIMIT);

        // Voice index here is not the same as in processing storage
        for voice_index in 0..num_active_voices {
            // Voice modulation input storage, indexed by operator
            let mut voice_modulation_inputs_left = [_mm256_setzero_pd(); 4];
            let mut voice_modulation_inputs_right = [_mm256_setzero_pd(); 4];

            let key_velocity_splat = _mm256_set1_pd(key_velocities[voice_index]);

            // Go through operators downwards, starting with operator 4
            for operator_index in 0..4 { // FIXME: better iterator with 3, 2, 1, 0 possible?
                let operator_index = 3 - operator_index;

                // FIXME
                // Possibly skip generation based on previous dependency analysis
                // if !operator_generate_audio[operator_index]{
                //     continue;
                // }

                let operator_modulation_target = operator_modulation_targets[operator_index];

                // Get panning as value between -1 and 1
                let pan_transformed = _mm256_mul_pd(
                    two_splat,
                    _mm256_sub_pd(
                        operator_panning[operator_index],
                        zero_point_five_splat
                    )
                );

                let pan_tendency_right = _mm256_max_pd(
                    pan_transformed,
                    zero_splat
                );
                let pan_tendency_left = _mm256_max_pd(
                    _mm256_mul_pd(
                        pan_transformed,
                        _mm256_set1_pd(-1.0) // FIXME - flip sign?
                    ),
                    zero_splat
                );

                if operator_wave_type[operator_index] == WaveType::WhiteNoise {
                    gen_noise_samples_for_voice_operator_channel( // left channel
                        &octasine.processing.rng,

                        &mut additive_outputs_left,
                        &mut voice_modulation_inputs_left[operator_modulation_target],

                        voice_envelope_volumes[voice_index][operator_index],
                        key_velocity_splat,

                        operator_volume[operator_index],
                        operator_additive[operator_index],

                        operator_constant_power_panning_left[operator_index]
                    );
                    gen_noise_samples_for_voice_operator_channel( // right channel
                        &octasine.processing.rng,

                        &mut additive_outputs_right,
                        &mut voice_modulation_inputs_right[operator_modulation_target],

                        voice_envelope_volumes[voice_index][operator_index],
                        key_velocity_splat,

                        operator_volume[operator_index],
                        operator_additive[operator_index],

                        operator_constant_power_panning_right[operator_index]
                    );
                } else {
                    gen_sin_samples_for_voice_operator_channel( // left channel
                        &mut additive_outputs_left,
                        &mut voice_modulation_inputs_left,
                        voice_modulation_inputs_right[operator_index],
                        &mut dummy_modulation_out,

                        voice_envelope_volumes[voice_index][operator_index],
                        voice_phases[voice_index][operator_index],
                        key_velocity_splat,

                        operator_index,
                        operator_volume[operator_index],
                        operator_additive[operator_index],
                        operator_modulation_target,
                        operator_feedback[operator_index],
                        operator_modulation_index[operator_index],

                        pan_tendency_left,
                        operator_constant_power_panning_left[operator_index]
                    );
                    gen_sin_samples_for_voice_operator_channel( // right channel
                        &mut additive_outputs_right,
                        &mut voice_modulation_inputs_right,
                        voice_modulation_inputs_left[operator_index],
                        &mut dummy_modulation_out,

                        voice_envelope_volumes[voice_index][operator_index],
                        voice_phases[voice_index][operator_index],
                        key_velocity_splat,

                        operator_index,
                        operator_volume[operator_index],
                        operator_additive[operator_index],
                        operator_modulation_target,
                        operator_feedback[operator_index],
                        operator_modulation_index[operator_index],

                        pan_tendency_right,
                        operator_constant_power_panning_right[operator_index]
                    );
                }
            } // End of operator iteration
        } // End of voice iteration

        // --- Summed additive outputs: apply master volume and hard limit.

        additive_outputs_left = _mm256_mul_pd(master_volume_factor, additive_outputs_left);
        additive_outputs_right = _mm256_mul_pd(master_volume_factor, additive_outputs_right);

        additive_outputs_left = hard_limit(additive_outputs_left);
        additive_outputs_right = hard_limit(additive_outputs_right);

        // --- Write additive outputs to audio buffer

        // Converting to f32s and writing directly would be nice..
        
        let mut tmp_left = [0.0f64; VECTOR_WIDTH];
        let mut tmp_right = [0.0f64; VECTOR_WIDTH];

        _mm256_storeu_pd(&mut tmp_left[0], additive_outputs_left);
        _mm256_storeu_pd(&mut tmp_right[0], additive_outputs_right);
        
        for (additive_out_left, additive_out_right, buffer_left, buffer_right) in izip!(
            tmp_left.iter(),
            tmp_right.iter(),
            audio_buffer_left_chunk.iter_mut(),
            audio_buffer_right_chunk.iter_mut()
        ){
            *buffer_left = *additive_out_left as f32;
            *buffer_right = *additive_out_right as f32;
        }

        // --- Clean up voice data for next pass

        voice_envelope_volumes.clear();
        voice_phases.clear();
        key_velocities.clear();

        #[cfg(feature = "with-coz")]
        coz::progress!();
    } // End of pass iteration
}


#[inline]
#[target_feature(enable = "avx")]
unsafe fn gen_noise_samples_for_voice_operator_channel(
    rng: &Rng,

    additive_outputs: &mut __m256d,
    modulation_outputs: &mut __m256d,

    envelope_volume: __m256d,
    key_velocity: __m256d,

    operator_volume: __m256d,
    operator_additive: __m256d,

    constant_power_panning: __m256d
){
    let one_splat = _mm256_set1_pd(1.0);
    let two_splat = _mm256_set1_pd(2.0);

    let samples = gen_noise_samples(
        rng,
        one_splat,
        two_splat
    );

    let volume_product = _mm256_mul_pd(operator_volume, envelope_volume);

    write_samples_to_outputs(
        volume_product,
        constant_power_panning,
        operator_additive,
        key_velocity,
        modulation_outputs,
        additive_outputs,
        samples,
    );
}


#[inline]
#[target_feature(enable = "avx")]
unsafe fn gen_sin_samples_for_voice_operator_channel(
    additive_outputs: &mut __m256d,
    voice_modulation_inputs_for_channel: &mut [__m256d; 4],
    modulation_in_for_other_channel: __m256d,
    dummy_modulation_out: &mut __m256d,

    envelope_volume: __m256d,
    voice_phases: __m256d,
    key_velocity: __m256d,

    operator_index: usize,
    operator_volume: __m256d,
    operator_additive: __m256d,
    operator_modulation_target: usize,
    operator_feedback: __m256d,
    operator_modulation_index: __m256d,

    pan_tendency: __m256d,
    constant_power_panning: __m256d
){
    let (
        voice_modulation_inputs_for_channel_below,
        voice_modulation_inputs_for_channel_above
    ) = voice_modulation_inputs_for_channel.split_at_mut(operator_index);

    let modulation_in_for_channel = voice_modulation_inputs_for_channel_above[0];

    let modulation_out = if operator_index == 0 {
        dummy_modulation_out
    } else {
        &mut voice_modulation_inputs_for_channel_below[operator_modulation_target]
    };

    let tau_splat = _mm256_set1_pd(TAU);
    let one_splat = _mm256_set1_pd(1.0);

    let one_minus_pan_tendency = _mm256_sub_pd(one_splat, pan_tendency);
    let volume_product = _mm256_mul_pd(operator_volume, envelope_volume);

    let samples = gen_sin_samples(
        tau_splat,
        voice_phases,
        modulation_in_for_channel,
        modulation_in_for_other_channel,
        pan_tendency,
        one_minus_pan_tendency,
        operator_feedback,
        operator_modulation_index,
    );

    write_samples_to_outputs(
        volume_product,
        constant_power_panning,
        operator_additive,
        key_velocity,
        modulation_out,
        additive_outputs,
        samples,
    );

    #[cfg(feature = "with-coz")]
    coz::progress!();
}


#[inline]
#[target_feature(enable = "avx")]
unsafe fn gen_sin_samples(
    tau_splat: __m256d,
    voice_phases: __m256d,
    modulation_in_for_channel: __m256d,
    modulation_in_for_other_channel: __m256d,
    pan_tendency: __m256d,
    one_minus_pan_tendency: __m256d,
    operator_feedback_splat: __m256d,
    operator_modulation_index_splat: __m256d,
) -> __m256d {
    let phase = _mm256_mul_pd(voice_phases, tau_splat);

    // Weird modulation input panning
    let modulation_in_channel_sum = _mm256_add_pd(
        modulation_in_for_channel,
        modulation_in_for_other_channel
    );
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
unsafe fn write_samples_to_outputs(
    volume_product: __m256d,
    constant_power_panning: __m256d,
    operator_additive: __m256d,
    key_velocity_splat: __m256d,
    modulation_out: &mut __m256d, // FIXME: return instead?
    additive_out: &mut __m256d, // FIXME: return instead?
    sample: __m256d,
){
    // Adjust sample volume
    let sample = _mm256_mul_pd(
        sample,
        _mm256_mul_pd(
            volume_product,
            constant_power_panning
        )
    );

    let additional_additive_out = _mm256_mul_pd(sample, operator_additive);
    let additional_modulation_out = _mm256_sub_pd(
        sample,
        additional_additive_out
    );

    *modulation_out = _mm256_add_pd(
        *modulation_out,
        additional_modulation_out
    );

    *additive_out = _mm256_add_pd(
        *additive_out,
        _mm256_mul_pd(
            additional_additive_out,
            key_velocity_splat
        )
    );
}


#[inline]
#[target_feature(enable = "avx")]
unsafe fn hard_limit(samples: __m256d) -> __m256d {
    let max_volume_splat = _mm256_set1_pd(5.0);
    let min_volume_splat = _mm256_set1_pd(-5.0);

    let samples = _mm256_min_pd(max_volume_splat, samples);
    let samples = _mm256_max_pd(min_volume_splat, samples);

    samples
}