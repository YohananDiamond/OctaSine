use iced_baseview::widget::text_input::{Appearance, StyleSheet};

use super::Theme;

impl StyleSheet for Theme {
    type Style = ();

    fn active(&self, _style: &Self::Style) -> Appearance {
        match self {
            Self::Dark => {
                use super::colors::dark::GRAY_300;

                Appearance {
                    background: GRAY_300.into(),
                    border_radius: 3.0,
                    border_width: 1.0,
                    border_color: GRAY_300,
                }
            }
            Self::Light => {
                use super::colors::light::{BORDER, SURFACE};

                Appearance {
                    background: SURFACE.into(),
                    border_radius: 3.0,
                    border_width: 1.0,
                    border_color: BORDER,
                }
            }
        }
    }

    fn focused(&self, style: &Self::Style) -> Appearance {
        self.active(style)
    }

    fn placeholder_color(&self, _style: &Self::Style) -> iced_baseview::Color {
        match self {
            Self::Dark => super::colors::dark::GRAY_800,
            Self::Light => super::colors::light::GRAY_300,
        }
    }

    fn value_color(&self, _style: &Self::Style) -> iced_baseview::Color {
        match self {
            Self::Dark => super::colors::dark::TEXT,
            Self::Light => super::colors::light::TEXT,
        }
    }

    fn selection_color(&self, _style: &Self::Style) -> iced_baseview::Color {
        match self {
            Self::Dark => super::colors::dark::GRAY_500,
            Self::Light => super::colors::light::GRAY_700,
        }
    }
}
