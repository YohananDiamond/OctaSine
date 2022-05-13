use crate::common::NUM_LFOS;
use crate::parameter_values::*;

use super::atomic_double::AtomicPositiveDouble;

pub struct PatchParameter {
    value: AtomicPositiveDouble,
    pub name: String,
    value_from_text: fn(String) -> Option<f64>,
    pub format: fn(f64) -> String,
}

impl PatchParameter {
    pub fn new<V: ParameterValue>(name: &str, default: V) -> Self {
        Self {
            name: name.to_string(),
            value: AtomicPositiveDouble::new(default.to_patch()),
            value_from_text: |v| V::new_from_text(v).map(|v| v.to_patch()),
            format: |v| V::new_from_patch(v).get_formatted(),
        }
    }

    pub fn set_value(&self, value: f64) {
        self.value.set(value);
    }

    pub fn get_value(&self) -> f64 {
        self.value.get()
    }

    pub fn get_value_text(&self) -> String {
        (self.format)(self.value.get())
    }

    pub fn set_from_text(&self, text: String) -> bool {
        if let Some(value) = (self.value_from_text)(text) {
            self.value.set(value);

            true
        } else {
            false
        }
    }
}

pub fn patch_parameters() -> Vec<PatchParameter> {
    let mut parameters = vec![
        master_volume(),
        master_frequency(),
        // Operator 1
        operator_volume(0),
        operator_active(0),
        operator_mix(0),
        operator_panning(0),
        operator_wave_type(0),
        operator_feedback(0),
        operator_frequency_ratio(0),
        operator_frequency_free(0),
        operator_frequency_fine(0),
        operator_attack_duration(0),
        operator_attack_volume(0),
        operator_decay_duration(0),
        operator_decay_volume(0),
        operator_release_duration(0),
        // Operator 2
        operator_volume(1),
        operator_active(1),
        operator_mix(1),
        operator_panning(1),
        operator_wave_type(1),
        operator_modulation_target_1(),
        operator_modulation_index(1),
        operator_feedback(1),
        operator_frequency_ratio(1),
        operator_frequency_free(1),
        operator_frequency_fine(1),
        operator_attack_duration(1),
        operator_attack_volume(1),
        operator_decay_duration(1),
        operator_decay_volume(1),
        operator_release_duration(1),
        // Operator 3
        operator_volume(2),
        operator_active(2),
        operator_mix(2),
        operator_panning(2),
        operator_wave_type(2),
        operator_modulation_target_2(),
        operator_modulation_index(2),
        operator_feedback(2),
        operator_frequency_ratio(2),
        operator_frequency_free(2),
        operator_frequency_fine(2),
        operator_attack_duration(2),
        operator_attack_volume(2),
        operator_decay_duration(2),
        operator_decay_volume(2),
        operator_release_duration(2),
        // Operator 4
        operator_volume(3),
        operator_active(3),
        operator_mix(3),
        operator_panning(3),
        operator_wave_type(3),
        operator_modulation_target_3(),
        operator_modulation_index(3),
        operator_feedback(3),
        operator_frequency_ratio(3),
        operator_frequency_free(3),
        operator_frequency_fine(3),
        operator_attack_duration(3),
        operator_attack_volume(3),
        operator_decay_duration(3),
        operator_decay_volume(3),
        operator_release_duration(3),
    ];

    for lfo_index in 0..NUM_LFOS {
        parameters.extend(vec![
            lfo_target_parameter(lfo_index),
            lfo_bpm_sync(lfo_index),
            lfo_frequency_ratio(lfo_index),
            lfo_frequency_free(lfo_index),
            lfo_mode(lfo_index),
            lfo_shape(lfo_index),
            lfo_amount(lfo_index),
            lfo_active(lfo_index),
        ])
    }

    parameters
}

fn master_volume() -> PatchParameter {
    PatchParameter::new("Master volume", MasterVolumeValue::default())
}

fn master_frequency() -> PatchParameter {
    PatchParameter::new("Master frequency", MasterFrequencyValue::default())
}

fn operator_volume(index: usize) -> PatchParameter {
    PatchParameter::new(
        &format!("OP {} vol", index + 1),
        OperatorVolumeValue::default(),
    )
}

fn operator_active(index: usize) -> PatchParameter {
    PatchParameter::new(
        &format!("OP {} active", index + 1),
        OperatorActiveValue::default(),
    )
}

fn operator_mix(index: usize) -> PatchParameter {
    PatchParameter::new(
        &format!("OP {} mix out", index + 1),
        OperatorMixOutValue::new(index),
    )
}

fn operator_panning(index: usize) -> PatchParameter {
    PatchParameter::new(
        &format!("OP {} pan", index + 1),
        OperatorPanningValue::default(),
    )
}

fn operator_frequency_ratio(index: usize) -> PatchParameter {
    PatchParameter::new(
        &format!("OP {} freq ratio", index + 1),
        OperatorFrequencyRatioValue::default(),
    )
}

fn operator_frequency_free(index: usize) -> PatchParameter {
    PatchParameter::new(
        &format!("OP {} freq free", index + 1),
        OperatorFrequencyFreeValue::default(),
    )
}

fn operator_frequency_fine(index: usize) -> PatchParameter {
    PatchParameter::new(
        &format!("OP {} freq fine", index + 1),
        OperatorFrequencyFineValue::default(),
    )
}

fn operator_feedback(index: usize) -> PatchParameter {
    PatchParameter::new(
        &format!("OP {} feedback", index + 1),
        OperatorFeedbackValue::default(),
    )
}

fn operator_modulation_index(index: usize) -> PatchParameter {
    PatchParameter::new(
        &format!("OP {} mod out", index + 1),
        OperatorModOutValue::default(),
    )
}

fn operator_wave_type(index: usize) -> PatchParameter {
    PatchParameter::new(
        &format!("OP {} wave", index + 1),
        OperatorWaveTypeValue::default(),
    )
}

fn operator_attack_duration(index: usize) -> PatchParameter {
    PatchParameter::new(
        &format!("OP {} attack time", index + 1),
        OperatorAttackDurationValue::default(),
    )
}

fn operator_attack_volume(index: usize) -> PatchParameter {
    PatchParameter::new(
        &format!("OP {} attack vol", index + 1),
        OperatorAttackVolumeValue::default(),
    )
}

fn operator_decay_duration(index: usize) -> PatchParameter {
    PatchParameter::new(
        &format!("OP {} decay time", index + 1),
        OperatorDecayDurationValue::default(),
    )
}

fn operator_decay_volume(index: usize) -> PatchParameter {
    PatchParameter::new(
        &format!("OP {} decay vol", index + 1),
        OperatorDecayVolumeValue::default(),
    )
}

fn operator_release_duration(index: usize) -> PatchParameter {
    PatchParameter::new(
        &format!("OP {} release time", index + 1),
        OperatorReleaseDurationValue::default(),
    )
}

fn operator_modulation_target_1() -> PatchParameter {
    PatchParameter::new("OP 2 mod target", Operator2ModulationTargetValue::default())
}

fn operator_modulation_target_2() -> PatchParameter {
    PatchParameter::new("OP 3 mod target", Operator3ModulationTargetValue::default())
}

fn operator_modulation_target_3() -> PatchParameter {
    PatchParameter::new("OP 4 mod target", Operator4ModulationTargetValue::default())
}

fn lfo_target_parameter(index: usize) -> PatchParameter {
    let title = format!("LFO {} target", index + 1);

    match index {
        0 => PatchParameter::new(&title, Lfo1TargetParameterValue::default()),
        1 => PatchParameter::new(&title, Lfo2TargetParameterValue::default()),
        2 => PatchParameter::new(&title, Lfo3TargetParameterValue::default()),
        3 => PatchParameter::new(&title, Lfo4TargetParameterValue::default()),
        _ => unreachable!(),
    }
}

fn lfo_shape(index: usize) -> PatchParameter {
    PatchParameter::new(
        &format!("LFO {} shape", index + 1),
        LfoShapeValue::default(),
    )
}

fn lfo_mode(index: usize) -> PatchParameter {
    PatchParameter::new(&format!("LFO {} mode", index + 1), LfoModeValue::default())
}

fn lfo_bpm_sync(index: usize) -> PatchParameter {
    PatchParameter::new(
        &format!("LFO {} bpm sync", index + 1),
        LfoBpmSyncValue::default(),
    )
}

fn lfo_frequency_ratio(index: usize) -> PatchParameter {
    PatchParameter::new(
        &format!("LFO {} freq ratio", index + 1),
        LfoFrequencyRatioValue::default(),
    )
}

fn lfo_frequency_free(index: usize) -> PatchParameter {
    PatchParameter::new(
        &format!("LFO {} freq free", index + 1),
        LfoFrequencyFreeValue::default(),
    )
}

fn lfo_amount(index: usize) -> PatchParameter {
    PatchParameter::new(
        &format!("LFO {} amount", index + 1),
        LfoAmountValue::default(),
    )
}

fn lfo_active(index: usize) -> PatchParameter {
    PatchParameter::new(
        &format!("LFO {} active", index + 1),
        LfoActiveValue::default(),
    )
}

#[cfg(test)]
mod tests {
    use crate::sync::change_info::MAX_NUM_PARAMETERS;

    use super::patch_parameters;

    #[test]
    fn test_sync_parameters_len() {
        assert!(patch_parameters().len() <= MAX_NUM_PARAMETERS);
    }
}
