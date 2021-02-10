use iced_baseview::{
    Container, Element, Text, Length, Align, Row, Rule, Space, HorizontalAlignment, Column, button, Button
};


use crate::GuiSyncHandle;
use crate::parameters::values::{
    OperatorAdditiveValue, OperatorFeedbackValue, OperatorFrequencyFineValue,
    OperatorFrequencyFreeValue, OperatorFrequencyRatioValue,
    OperatorModulationIndexValue, OperatorPanningValue, OperatorVolumeValue,
    OperatorWaveTypeValue
};

use super::{FONT_SIZE, FONT_BOLD, FONT_VERY_BOLD, LINE_HEIGHT, Message};
use super::envelope::Envelope;
use super::knob::{self, OctaSineKnob};
use super::boolean_picker::{self, BooleanPicker};


pub struct OperatorWidgets {
    index: usize,
    pub volume: OctaSineKnob<OperatorVolumeValue>,
    pub panning: OctaSineKnob<OperatorPanningValue>,
    pub wave_type: BooleanPicker<OperatorWaveTypeValue>,
    pub mod_index: OctaSineKnob<OperatorModulationIndexValue>,
    pub feedback: OctaSineKnob<OperatorFeedbackValue>,
    pub frequency_ratio: OctaSineKnob<OperatorFrequencyRatioValue>,
    pub frequency_free: OctaSineKnob<OperatorFrequencyFreeValue>,
    pub frequency_fine: OctaSineKnob<OperatorFrequencyFineValue>,
    pub additive: Option<OctaSineKnob<OperatorAdditiveValue>>,
    pub envelope: Envelope,
    pub zoom_in: button::State,
    pub zoom_out: button::State,
    pub sync_viewport: button::State,
    pub zoom_to_fit: button::State,
}


impl OperatorWidgets {
    pub fn new<H: GuiSyncHandle>(
        sync_handle: &H,
        operator_index: usize,
    ) -> Self {
        let (volume, panning, wave, additive, mod_index, feedback, ratio, free, fine) = match operator_index {
            0 => ( 2,  3,  4,  0,  5,  6,  7,  8,  9),
            1 => (15, 16, 17, 18, 19, 20, 21, 22, 23),
            2 => (29, 30, 31, 32, 34, 35, 36, 37, 38),
            3 => (44, 45, 46, 47, 49, 50, 51, 52, 53),
            _ => unreachable!(),
        };

        let additive_knob = if operator_index == 0 {
            None
        } else {
            Some(knob::operator_additive(sync_handle, additive))
        };

        Self {
            index: operator_index,
            volume: knob::operator_volume(sync_handle, volume, operator_index),
            panning: knob::operator_panning(sync_handle, panning),
            wave_type: boolean_picker::wave_type(sync_handle, wave),
            mod_index: knob::operator_mod_index(sync_handle, mod_index),
            feedback: knob::operator_feedback(sync_handle, feedback),
            frequency_ratio: knob::operator_frequency_ratio(sync_handle, ratio),
            frequency_free: knob::operator_frequency_free(sync_handle, free),
            frequency_fine: knob::operator_frequency_fine(sync_handle, fine),
            additive: additive_knob,
            envelope: Envelope::new(sync_handle, operator_index),
            zoom_in: button::State::default(),
            zoom_out: button::State::default(),
            sync_viewport: button::State::default(),
            zoom_to_fit: button::State::default(),
        }
    }

    pub fn view(&mut self) -> Element<Message> {
        let operator_number = Text::new(format!("{}", self.index + 1))
            .size(FONT_SIZE * 2)
            .font(FONT_VERY_BOLD)
            .horizontal_alignment(HorizontalAlignment::Center);

        let mut row = Row::new()
            .push(
                Container::new(operator_number)
                    .width(Length::Units(LINE_HEIGHT * 4))
                    .height(Length::Units(LINE_HEIGHT * 6))
                    .align_x(Align::Center)
                    .align_y(Align::Center)
            )
            // .push(Space::with_width(Length::Units(LINE_HEIGHT)))
            .push(self.wave_type.view())
            .push(self.volume.view())
            .push(self.panning.view());
        
        if let Some(additive) = self.additive.as_mut() {
            row = row.push(additive.view())
        } else {
            row = row.push(Space::with_width(Length::Units(LINE_HEIGHT * 4)))
        }

        row = row
            .push(
                Container::new(
                    Rule::vertical(LINE_HEIGHT)
                )
                    .height(Length::Units(LINE_HEIGHT * 6))
            )
            .push(self.mod_index.view())
            .push(self.feedback.view());
        
        row = row
            .push(
                Container::new(
                    Rule::vertical(LINE_HEIGHT)
                )
                    .height(Length::Units(LINE_HEIGHT * 6)))
            .push(self.frequency_ratio.view())
            .push(self.frequency_free.view())
            .push(self.frequency_fine.view());
        
        let sync_viewports_message = Message::EnvelopeSyncViewports {
            viewport_factor: self.envelope.get_viewport_factor(),
            x_offset: self.envelope.get_x_offset(),
        };
        let zoom_to_fit_message = Message::EnvelopeZoomToFit(self.index);
        
        row = row
            .push(
                Container::new(
                    Rule::vertical(LINE_HEIGHT)
                )
                    .height(Length::Units(LINE_HEIGHT * 6))
            )
            .push(
                Column::new()
                    .push(self.envelope.view())
            )
            .push(
                Column::new()
                    .width(Length::Units(LINE_HEIGHT * 3))
                    .align_items(Align::End)
                    .push(
                        Row::new()
                            .push(
                                Button::new(&mut self.zoom_out, Text::new("−").font(FONT_VERY_BOLD))
                                    .on_press(Message::EnvelopeZoomOut(self.index))
                            )
                            .push(
                                Space::with_width(Length::Units(3))
                            )
                            .push(
                                Button::new(&mut self.zoom_in, Text::new("+").font(FONT_VERY_BOLD))
                                    .on_press(Message::EnvelopeZoomIn(self.index))
                            )
                    )
                    .push(Space::with_height(Length::Units(LINE_HEIGHT * 1 - 10)))
                    .push(
                        Row::new()
                            .push(
                                Button::new(&mut self.zoom_to_fit, Text::new("FIT"))
                                    .on_press(zoom_to_fit_message)
                            )
                    )
                    .push(Space::with_height(Length::Units(LINE_HEIGHT * 1 - 10)))
                    .push(
                        Row::new()
                            .push(
                                Button::new(&mut self.sync_viewport, Text::new("DIST"))
                                    .on_press(sync_viewports_message)
                            )
                    )
            );

        row.into()
    }
}
