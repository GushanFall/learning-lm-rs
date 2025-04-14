mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::{option, path::PathBuf};
use egui::{response, TextWrapMode};
use tokenizers::Tokenizer;
use crate::kvcache::KVCache;

use eframe::egui;

// 故事续写（每次都是新的 KV Cache）
fn continue_story(input: &str) {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    // 加载模型
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    // 加载tokenizer
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    // 编码 input
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    // 创建新的 KV Cache
    let mut cache = llama.new_cache();
    // 生成输出
    let output_ids = llama.generate(
        &mut cache,
        input_ids,
        500,
        0.8,
        30,
        1.,
    );
    let output = tokenizer.decode(&output_ids, true).unwrap();
    // 打印输出
    print!("\n{}", input);
    println!("{}", output.replace("<|end_story|>", ""));
}


// 对话
pub struct Chat {
    llama: model::Llama<f32>,
    tokenizer: Tokenizer,
    max_len: usize,
    top_p: f32,
    top_k: u32,
    temperature: f32,
    messages: String,
    cache: KVCache<f32>,
}

impl Chat {
    // 创建 Chat 
    pub fn new() -> Self {
        let project_dir = env!("CARGO_MANIFEST_DIR");
        let model_dir = PathBuf::from(project_dir).join("models").join("chat");
        let system_message = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
        let llama = model::Llama::<f32>::from_safetensors(&model_dir);
        let mut cache = llama.new_cache();
        
        Self {
            llama,
            tokenizer: Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap(),
            max_len: 500,
            top_p: 0.8,
            top_k: 30,
            temperature: 1.,
            messages: system_message.to_string(),
            cache,
        }
    }

    // 修改生成参数
    pub fn set_max_len(&mut self, max_len: usize) {
        if max_len < 1 {
            panic!("max_len must be greater than 0");
        }
        self.max_len = max_len;
    }

    pub fn set_top_p(&mut self, top_p: f32) {
        if top_p < 0.0 || top_p > 1.0 {
            panic!("top_p must be between 0 and 1");
        }
        self.top_p = top_p;
    }

    pub fn set_top_k(&mut self, top_k: u32) {
        if top_k < 1 {
            panic!("top_k must be greater than 0");
        }
        self.top_k = top_k;
    }

    pub fn set_temperature(&mut self, temperature: f32) {
        if temperature < 0.0 || temperature > 1.0 {
            panic!("temperature must be between 0 and 1");
        }
        self.temperature = temperature;
    }

    // 创建新的 KV Cache
    pub fn new_cache(&mut self) {
        self.cache = self.llama.new_cache();
    }

    // 模型回复
    pub fn response(
        &mut self,
        user_input: &str,
    ) -> String {
        let user_message = format!("<|im_start|>user\n{}<|im_end|>\n", user_input);
        let assistant_message = "<|im_start|>assistant\n";
        let input = format!("{}{}{}", self.messages, user_message, assistant_message);
        // 编码 input
        let binding = self.tokenizer.encode(input, true).unwrap();
        let input_ids = binding.get_ids();
        // 生成输出
        let output_ids = self.llama.generate(
            &mut self.cache,
            input_ids,
            self.max_len,
            self.top_p,
            self.top_k,
            self.temperature,
        );
        let output = self.tokenizer.decode(&output_ids, true).unwrap();

        output
    }

    // 将模型输出合并到历史消息中
    pub fn merge_messages(&mut self, response: String) {
        self.messages = format!("{}{}{}", self.messages, response, "<|im_end|>\n");
    }
}

fn app() {
    let mut chat = Chat::new();
    loop {
        let mut input = String::new();
        println!("-----------------------------------------------------------");
        println!("User:");
        std::io::stdin().read_line(&mut input).unwrap();
        if input.trim() == "exit" {
            println!("Exit!");
            break;
        }else if input.trim().is_empty() {
            continue;
        }else if input.trim() == "clear" {
            chat.new_cache();
            println!("Cache cleared!");
            continue;
        }
        let response = chat.response(input.trim());
        println!("-----------------------------------------------------------");
        println!("Assistant:\n{}", response);
        chat.merge_messages(response.clone());
    }
}

pub struct AppGui {
    chat: Chat,
    user_input: String,
    messages: Vec<(String, bool)>, // (true for user, false for assistant)
    max_len: usize,
    top_p: f32,
    top_k: u32,
    temperature: f32,
}

impl AppGui {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let chat = Chat::new();
        let user_input = String::new();
        let messages = vec![
            ("Hello, I am a Chat Bot. You can chat with me.".to_string(), false),
        ];
        Self {
            chat,
            user_input,
            messages,
            max_len: 500,
            top_p: 0.8,
            top_k: 30,
            temperature: 1.0,
        }
    }

    fn send_message(&mut self) {
        if !self.user_input.is_empty() {
            // 设置参数
            self.chat.set_max_len(self.max_len);
            self.chat.set_top_p(self.top_p);
            self.chat.set_top_k(self.top_k);
            self.chat.set_temperature(self.temperature);
            // 发送消息
            self.messages.push((self.user_input.clone(), true));
            let response = self.chat.response(self.user_input.trim());
            // let response = "Hello, I am your assistant! You can tell ask me any problem, and i'll answer it. Don't be shy! :)".to_string();
            self.messages.push((response, false));
            self.user_input.clear();
        }
    }
}

impl eframe::App for AppGui {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 使用合并样式的方式而不是完全覆盖
        let mut style = (*ctx.style()).clone();
        
        // 只修改需要的文本样式
        style.text_styles.insert(
            egui::TextStyle::Heading, 
            egui::FontId::new(20.0, egui::FontFamily::Proportional)
        );
        
        // 设置基础字体（影响Body等默认样式）
        style.text_styles.insert(
            egui::TextStyle::Body, 
            egui::FontId::new(14.0, egui::FontFamily::Proportional)
        );
        
        // 设置按钮文字样式
        style.text_styles.insert(
            egui::TextStyle::Button, 
            egui::FontId::new(14.0, egui::FontFamily::Proportional)
        );
        
        ctx.set_style(style);

        egui::CentralPanel::default().show(ctx, |ui| {
            // 标题样式
            ui.vertical_centered(|ui| {
                ui.add_space(10.0);
                ui.heading(egui::RichText::new("Chat Assistant").color(egui::Color32::from_rgb(70, 130, 180)));
                ui.add_space(10.0);
            });

            // 按钮行布局
            ui.horizontal(|ui| {
                ui.add_space(10.0);
                if ui
                    .add(
                        egui::Button::new(egui::RichText::new("New Chat").size(14.0))
                            .fill(egui::Color32::from_rgb(70, 130, 180))
                            .min_size(egui::vec2(100.0, 30.0)),
                    )
                    .clicked()
                {
                    self.chat.new_cache();
                    self.messages.clear();
                    self.user_input.clear();
                };
                ui.add_space(10.0);
                ui.add(egui::Slider::new(&mut self.max_len, 1..=2000).text("Max Length"));
                ui.add_space(10.0);
                ui.add(egui::Slider::new(&mut self.top_p, 0.0..=1.0).text("Top P"));
                ui.add_space(10.0);
                ui.add(egui::Slider::new(&mut self.top_k, 1..=100).text("Top K"));
                ui.add_space(10.0);
                ui.add(egui::Slider::new(&mut self.temperature, 0.0..=1.0).text("Temperature"));
                ui.add_space(10.0);
            });

            ui.separator();

            // 消息区域
            let mut scroll_area = egui::ScrollArea::vertical();
            if !self.messages.is_empty() {
                scroll_area = scroll_area
                   .min_scrolled_height(ui.available_height() * 0.8)
                   .max_height(ui.available_height() * 0.8);
            }
            scroll_area
               .stick_to_bottom(true)
               .show(ui, |ui| {
                    ui.set_width(ui.available_width() - 10.0); // 右边留白
                    for (message, is_user) in &self.messages {
                        ui.with_layout(
                            if *is_user {
                                egui::Layout::right_to_left(egui::Align::BOTTOM)
                            } else {
                                egui::Layout::left_to_right(egui::Align::BOTTOM)
                            },
                            |ui| {
                                let bubble_color = if *is_user {
                                    egui::Color32::from_rgb(70, 130, 180)
                                } else {
                                    egui::Color32::from_rgb(220, 220, 220)
                                };
                                
                                egui::Frame::group(ui.style())
                                   .fill(bubble_color)
                                   .corner_radius(8.0)
                                   .inner_margin(egui::vec2(12.0, 8.0))
                                   .show(ui, |ui| {
                                        ui.add(egui::Label::new(
                                            egui::RichText::new(format!(
                                                "{}",
                                                message
                                            ))
                                           .color(if *is_user {
                                                egui::Color32::WHITE
                                            } else {
                                                egui::Color32::BLACK
                                            }),
                                        ).wrap());
                                    });
                                ui.add_space(8.0);
                            },
                        );
                        ui.add_space(8.0);
                    }
                });

            ui.separator();

            // 输入区域
            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(10.0);
                    egui::Frame::group(ui.style())
                       .fill(egui::Color32::from_rgb(245, 245, 245))
                       .inner_margin(egui::vec2(10.0, 8.0))
                       .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                let text_edit = egui::TextEdit::singleline(&mut self.user_input)
                                   .desired_width(ui.available_width() - 100.0)
                                   .text_color(egui::Color32::BLACK)
                                   .font(egui::FontId::new(14.0, egui::FontFamily::Proportional));
                                
                                if ui.add(text_edit).lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                    self.send_message();
                                }

                                if ui
                                   .add(
                                        egui::Button::new(egui::RichText::new("Send").size(14.0))
                                           .fill(egui::Color32::from_rgb(70, 130, 180))
                                           .min_size(egui::vec2(80.0, 32.0)),
                                    )
                                   .clicked()
                                {
                                    self.send_message();
                                }
                            });
                        });
                    ui.add_space(10.0);
                });
            });
        });
    }
}

fn main() {
    // 故事续写
    // continue_story("Once upon a time");
    // 命令行对话
    // app();
    // GUI 对话
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 600.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Chat",
        options,
        Box::new(|cc| Ok(Box::new(AppGui::new(cc)))),
    ).unwrap();
}
