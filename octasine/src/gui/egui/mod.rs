use egui::{CtxRef, Window};
use egui_baseview::Queue;

use crate::GuiSyncHandle;
use crate::constants::PLUGIN_NAME;

pub fn build<H: GuiSyncHandle>(_egui_ctx: &CtxRef, _queue: &mut Queue, _sync_handle: &mut H){

}

pub fn update<H: GuiSyncHandle>(egui_ctx: &CtxRef, _queue: &mut Queue, sync_handle: &mut H){
    Window::new("Operator 1")
	.fixed_pos(&(0.0, 0.0))
	.fixed_size(&(10.0, 100.0))
	.resizable(false)
	.collapsible(false)
    	.show(&egui_ctx, |ui| {
	    let mut v = sync_handle.get_parameter(0);
	    if ui.add(egui::Slider::new(&mut v, 0.0..=1.0)).changed() {
		sync_handle.set_parameter(0, v);
	    }
	});
}