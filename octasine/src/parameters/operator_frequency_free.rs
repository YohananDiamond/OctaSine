use compact_str::format_compact;
use compact_str::CompactString;

use super::utils::*;
use super::ParameterValue;

const OPERATOR_FREE_STEPS: &[f32] = &[
    1.0 / 1024.0,
    1.0 / 64.0,
    1.0 / 16.0,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    4.0,
    16.0,
    64.0,
    1024.0,
];

#[derive(Debug, Clone, Copy)]
pub struct OperatorFrequencyFreeValue(f64);

impl Default for OperatorFrequencyFreeValue {
    fn default() -> Self {
        Self(1.0)
    }
}

impl ParameterValue for OperatorFrequencyFreeValue {
    type Value = f64;

    fn new_from_audio(value: Self::Value) -> Self {
        Self(value)
    }
    fn new_from_text(text: &str) -> Option<Self> {
        const MIN: f32 = OPERATOR_FREE_STEPS[0];
        const MAX: f32 = OPERATOR_FREE_STEPS[OPERATOR_FREE_STEPS.len() - 1];

        parse_valid_f32(text, MIN, MAX).map(|v| Self(v.into()))
    }
    fn get(self) -> Self::Value {
        self.0
    }
    fn new_from_patch(value: f32) -> Self {
        Self(map_patch_to_audio_value_with_steps(&OPERATOR_FREE_STEPS, value) as f64)
    }
    fn to_patch(self) -> f32 {
        map_audio_to_patch_value_with_steps(&OPERATOR_FREE_STEPS, self.0 as f32)
    }
    fn get_formatted(self) -> CompactString {
        format_compact!("{:.04}", self.0)
    }
}
