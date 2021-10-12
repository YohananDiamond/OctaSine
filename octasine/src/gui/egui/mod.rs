use egui::{CtxRef, FontDefinitions, FontFamily, TextStyle, Window, Ui};
use egui_baseview::Queue;

use crate::GuiSyncHandle;

const OPEN_SANS_REGULAR: &[u8] =
    include_bytes!("../../../../contrib/open-sans/OpenSans-Regular.ttf");
const OPEN_SANS_SEMI_BOLD: &[u8] =
    include_bytes!("../../../../contrib/open-sans/OpenSans-SemiBold.ttf");
const OPEN_SANS_BOLD: &[u8] = include_bytes!("../../../../contrib/open-sans/OpenSans-Bold.ttf");

pub fn build<H: GuiSyncHandle>(ctx: &CtxRef, _queue: &mut Queue, _sync_handle: &mut H){
    let mut fonts = FontDefinitions::default();

    fonts.font_data.insert("Open Sans Regular".to_owned(), ::std::borrow::Cow::Borrowed(OPEN_SANS_REGULAR));
    fonts.font_data.insert("Open Sans Semi Bold".to_owned(), ::std::borrow::Cow::Borrowed(OPEN_SANS_SEMI_BOLD));
    fonts.font_data.insert("Open Sans Bold".to_owned(), ::std::borrow::Cow::Borrowed(OPEN_SANS_BOLD));

    fonts.fonts_for_family.get_mut(&FontFamily::Proportional).unwrap().insert(0, "Open Sans Regular".to_owned());
    fonts.fonts_for_family.get_mut(&FontFamily::Monospace).unwrap().insert(0, "Open Sans Regular".to_owned());

    // ctx.set_fonts(fonts); // Blurry

    let mut style: egui::Style = (*ctx.style()).clone();
    style.body_text_style = TextStyle::Small;
    style.override_text_style = Some(TextStyle::Small);
    ctx.set_style(style);
}

pub fn update<H: GuiSyncHandle>(egui_ctx: &CtxRef, _queue: &mut Queue, sync_handle: &mut H){
    Window::new("WAVE")
	.fixed_pos(&(0.0, 0.0))
	.fixed_size(&(500.0, 400.0))
	.resizable(false)
	.collapsible(false)
	.title_bar(false)
    	.show(&egui_ctx, |ui| {
	    ui.horizontal(|ui| {
		ui.label("1");

		ui.group(|ui| {
		    ui.label("WAVE");

		    let mut v = sync_handle.get_parameter(0);
		    if ui.add(egui::Slider::new(&mut v, 0.0..=1.0)).changed() {
			    sync_handle.set_parameter(0, v);
		    }
		});

		titled_drag_value(sync_handle, ui, "VOLUME");
		titled_drag_value(sync_handle, ui, "VOLUME");
	    });
	});
}

fn titled_drag_value<H: GuiSyncHandle>(sync_handle: &H, ui: &mut Ui, title: &str) {
    ui.group(|ui| {
	ui.vertical_centered_justified(|ui| {
	    ui.label(title);

	    let drag_value = egui::DragValue::from_get_set(|opt_new_value| {
		    if let Some(v) = opt_new_value {
			sync_handle.set_parameter(0, v);
		    }
		    sync_handle.get_parameter(0)
		})
		.fixed_decimals(3)
		.clamp_range(0.0..=1.0)
		.speed(0.01);

	    ui.add(drag_value);
	})
    });
}