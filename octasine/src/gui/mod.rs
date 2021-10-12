use cfg_if::cfg_if;

use baseview::{Size, WindowOpenOptions, WindowScalePolicy};
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use serde::{Deserialize, Serialize};
use vst::editor::Editor;

use super::GuiSyncHandle;
use crate::constants::PLUGIN_NAME;

cfg_if! {
    if #[cfg(feature = "iced_gui")] {
        mod iced;

        use iced::OctaSineIcedApplication;

        pub struct GuiSettings {
            pub theme: iced::style::Theme,
        }
    } else {
        mod egui;
    }
}

pub const GUI_WIDTH: usize = 12 * 66;
pub const GUI_HEIGHT: usize = 12 * 61;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]

pub struct Gui<H: GuiSyncHandle> {
    sync_state: H,
    opened: bool,
}

impl<H: GuiSyncHandle> Gui<H> {
    pub fn new(sync_state: H) -> Self {
        Self {
            sync_state,
            opened: false,
        }
    }
    
    cfg_if!{
        if #[cfg(feature = "iced_gui")] {
            fn get_iced_baseview_settings(sync_handle: H) -> iced_baseview::Settings<H> {
                iced_baseview::Settings {
                    window: WindowOpenOptions {
                        size: Size::new(GUI_WIDTH as f64, GUI_HEIGHT as f64),
                        scale: WindowScalePolicy::SystemScaleFactor,
                        title: PLUGIN_NAME.to_string(),
                    },
                    flags: sync_handle,
                }
            }

            pub fn open_parented(parent: ParentWindow, sync_handle: H) {
                iced_baseview::IcedWindow::<OctaSineIcedApplication<_>>::open_parented(
                    &parent,
                    Self::get_iced_baseview_settings(sync_handle),
                );
            }

            pub fn open_blocking(sync_handle: H) {
                let settings = Self::get_iced_baseview_settings(sync_handle);

                iced_baseview::IcedWindow::<OctaSineIcedApplication<_>>::open_blocking(settings);
            }
        } else {
            fn get_egui_baseview_settings() -> egui_baseview::Settings {
                egui_baseview::Settings {
                    window: WindowOpenOptions {
                        size: Size::new(GUI_WIDTH as f64, GUI_HEIGHT as f64),
                        scale: WindowScalePolicy::SystemScaleFactor,
                        title: PLUGIN_NAME.to_string(),
                    },
                    render_settings: Default::default(),
                }
            }

            pub fn open_parented(parent: ParentWindow, sync_handle: H) {
                let settings = Self::get_egui_baseview_settings();

                egui_baseview::EguiWindow::open_parented(&parent, settings, sync_handle, egui::build, egui::update);
            }

            pub fn open_blocking(sync_handle: H) {
                let settings = Self::get_egui_baseview_settings();

                egui_baseview::EguiWindow::open_blocking(settings, sync_handle, egui::build, egui::update);
            }
        }
    }
}

impl<H: GuiSyncHandle> Editor for Gui<H> {
    fn size(&self) -> (i32, i32) {
        (GUI_WIDTH as i32, GUI_HEIGHT as i32)
    }

    fn position(&self) -> (i32, i32) {
        (0, 0)
    }

    fn open(&mut self, parent: *mut ::core::ffi::c_void) -> bool {
        if self.opened {
            return false;
        }

        Self::open_parented(ParentWindow(parent), self.sync_state.clone());

        true
    }

    fn close(&mut self) {
        self.opened = false;
    }

    fn is_open(&mut self) -> bool {
        self.opened
    }
}

pub struct ParentWindow(pub *mut ::core::ffi::c_void);

unsafe impl HasRawWindowHandle for ParentWindow {
    #[cfg(target_os = "macos")]
    fn raw_window_handle(&self) -> RawWindowHandle {
        use raw_window_handle::macos::MacOSHandle;

        RawWindowHandle::MacOS(MacOSHandle {
            ns_view: self.0,
            ..MacOSHandle::empty()
        })
    }

    #[cfg(target_os = "windows")]
    fn raw_window_handle(&self) -> RawWindowHandle {
        use raw_window_handle::windows::WindowsHandle;

        RawWindowHandle::Windows(WindowsHandle {
            hwnd: self.0,
            ..WindowsHandle::empty()
        })
    }

    #[cfg(target_os = "linux")]
    fn raw_window_handle(&self) -> RawWindowHandle {
        use raw_window_handle::unix::XcbHandle;

        RawWindowHandle::Xcb(XcbHandle {
            window: self.0 as u32,
            ..XcbHandle::empty()
        })
    }
}
