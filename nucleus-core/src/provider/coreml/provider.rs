//! CoreML provider implementation for macOS.
//!
//! This module provides inference using Apple's CoreML framework.
//! Only available on macOS with the `coreml` feature enabled.

use crate::models::EmbeddingModel;
use crate::provider::{ChatRequest, ChatResponse, Message, Provider, ProviderError, Result};
use crate::Config;
use async_trait::async_trait;
use nucleus_plugin::PluginRegistry;
use std::ffi::{c_char, c_float, c_int, c_void, CString};
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tracing::{debug, info};

#[repr(C)]
struct CoreMLModelRef(*mut c_void);

#[repr(C)]
struct CoreMLStateRef(*mut c_void);

unsafe impl Send for CoreMLModelRef {}
unsafe impl Sync for CoreMLModelRef {}

unsafe impl Send for CoreMLStateRef {}
unsafe impl Sync for CoreMLStateRef {}

extern "C" {
    /// Loads a CoreML model from disk and returns an opaque handle.
    fn coreml_load_model(model_path: *const c_char) -> *mut c_void;

    /// Runs inference on a model using multi-array inputs/outputs.
    /// Returns 0 on success, non-zero on error.
    fn coreml_predict_multiarray(
        model: *mut c_void,
        input_name: *const c_char,
        input_data: *const c_float,
        input_size: usize,
        output_name: *const c_char,
        output_data: *mut c_float,
        output_size: usize,
    ) -> c_int;

    /// Creates a new MLState object for stateful sequence predictions.
    fn coreml_make_state(model: *mut c_void) -> *mut c_void;
    /// Frees an MLState object.
    fn coreml_free_state(state: *mut c_void);

    /// Runs stateful inference with KV cache for autoregressive generation.
    /// Returns 0 on success, non-zero on error.
    fn coreml_predict_stateful(
        model: *mut c_void,
        state: *mut c_void,
        input_ids: *const i32,
        input_ids_size: usize,
        causal_mask: *const c_float,
        mask_size: usize,
        output_data: *mut c_float,
        output_size: usize,
    ) -> c_int;

    /// Frees a CoreML model handle.
    fn coreml_free_model(model: *mut c_void);

    /// Queries the shape of a model input by name.
    /// Returns number of dimensions on success, negative on error.
    fn coreml_get_input_shape(
        model: *mut c_void,
        input_name: *const c_char,
        shape_out: *mut i64,
        max_dims: usize,
    ) -> c_int;
}

/// Returns the index of the maximum value in logits (greedy sampling).
fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

/// Encodes text as Unicode codepoints clamped to 127999 (fallback tokenizer).
fn simple_encode(text: &str) -> Vec<u32> {
    text.chars().map(|c| (c as u32).min(127_999)).collect()
}

/// Decodes token IDs back to text by treating them as Unicode codepoints.
fn simple_decode(tokens: &[u32]) -> String {
    tokens
        .iter()
        .filter_map(|&id| {
            if id < 128_000 {
                char::from_u32(id)
            } else {
                None
            }
        })
        .collect()
}

/// Creates a lower-triangular attention mask for autoregressive generation.
fn create_causal_mask(seq_len: usize) -> Vec<f32> {
    let mut mask = vec![f32::NEG_INFINITY; seq_len * seq_len];

    for i in 0..seq_len {
        for j in 0..=i {
            mask[i * seq_len + j] = 0.0;
        }
    }

    mask
}

/// Samples a token from logits using softmax with temperature scaling.
fn sample_with_temperature(logits: &[f32], temperature: f64) -> u32 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hash, Hasher};
    use std::time::SystemTime;

    let temp = temperature as f32;

    let max_logit = logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    let exp_logits: Vec<f32> = logits
        .iter()
        .map(|&logit| ((logit - max_logit) / temp).exp())
        .collect();

    let sum: f32 = exp_logits.iter().sum();
    let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum).collect();

    let mut hasher = RandomState::new().build_hasher();
    SystemTime::now().hash(&mut hasher);
    let seed = hasher.finish();
    let random_val = ((seed as f64) / (u64::MAX as f64)) as f32;

    let mut cumsum = 0.0;
    for (i, &prob) in probs.iter().enumerate() {
        cumsum += prob;
        if random_val < cumsum {
            return i as u32;
        }
    }

    (probs.len() - 1) as u32
}

pub struct CoreMLProvider {
    model: CoreMLModelRef,
    state: Option<CoreMLStateRef>,
    model_path: String,
    input_name: String,
    output_name: String,
    _registry: Arc<PluginRegistry>,
    _config: Config,
    _tokenizer: Option<Tokenizer>,
    _vocab_size: usize,
    #[allow(unused)]
    max_cache_length: usize,
}

impl CoreMLProvider {
    /// Creates a new CoreML provider.
    ///
    /// Loads the CoreML model from the configured path.
    pub async fn new(config: &Config, registry: Arc<PluginRegistry>) -> Result<Arc<Self>> {
        info!("CoreML provider initialized with Apple Neural Engine acceleration");

        let model_path = config.llm.model.clone();

        let expanded_path = if model_path.starts_with('~') {
            let home = std::env::var("HOME")
                .map_err(|_| ProviderError::Other("HOME environment variable not set".to_string()))?;
            model_path.replacen('~', &home, 1)
        } else {
            model_path.clone()
        };

        let path = Path::new(&expanded_path);
        info!("Searching for model at path {:?}", path.to_str());

        if !path.exists() {
            return Err(ProviderError::Other(format!(
                "CoreML model not found: {}",
                path.display()
            )));
        }

        let path_str = path
            .to_str()
            .ok_or_else(|| ProviderError::Other("Invalid UTF-8 in model path".to_string()))?;

        let c_path = CString::new(path_str)
            .map_err(|e| ProviderError::Other(format!("Invalid path: {}", e)))?;

        let handle = unsafe { coreml_load_model(c_path.as_ptr()) };

        if handle.is_null() {
            return Err(ProviderError::Other(
                "Failed to load CoreML model".to_string(),
            ));
        }

        // Create an MLState for sequence predictions, matching makeState().
        let state_handle = unsafe { coreml_make_state(handle) };
        let state = if state_handle.is_null() {
            None
        } else {
            Some(CoreMLStateRef(state_handle))
        };

        let tokenizer_parent = Path::new(&expanded_path)
            .parent()
            .ok_or_else(|| ProviderError::Other("Invalid model path".to_string()))?;
        let tokenizer_path = tokenizer_parent.join("tokenizer.json");

        let tokenizer = if tokenizer_path.exists() {
            let tok = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| ProviderError::Other(format!("Tokenizer load failed: {}", e)))?;

            let vocab_size = tok.get_vocab_size(false);
            info!("Tokenizer loaded: {} tokens", vocab_size);

            Some(tok)
        } else {
            info!(
                "No tokenizer.json found at {:?}, using default vocab size from config",
                tokenizer_path
            );
            None
        };

        let vocab_size = tokenizer
            .as_ref()
            .map(|t| t.get_vocab_size(false))
            .unwrap_or_else(|| {
                debug!("Using context_length 128256 as default vocab size");
                128_256
            });

        info!("CoreML model loaded: {}", path.display());

        Ok(Arc::new(Self {
            model: CoreMLModelRef(handle),
            state,
            model_path: path_str.to_string(),
            input_name: "inputIds".to_string(),
            output_name: "logits".to_string(),
            _registry: registry,
            _config: config.clone(),
            _tokenizer: tokenizer,
            _vocab_size: vocab_size,
            max_cache_length: 2048,
        }))
    }

    fn format_chat_prompt(&self, messages: &[Message]) -> Result<(String, Vec<u32>)> {
        if let Some(ref tokenizer) = self._tokenizer {
            let mut prompt = String::new();

            for message in messages {
                match message.role.as_str() {
                    "system" => {
                        prompt.push_str(
                            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
                        );
                        if let Some(ref context) = message.context {
                            prompt.push_str(context);
                            prompt.push_str("\n\n");
                        }
                        prompt.push_str(&message.content);
                        prompt.push_str("<|eot_id|>");
                    }
                    "user" => {
                        prompt
                            .push_str("<|start_header_id|>user<|end_header_id|>\n\n");
                        if let Some(ref context) = message.context {
                            prompt.push_str(context);
                            prompt.push_str("\n\n");
                        }
                        prompt.push_str(&message.content);
                        prompt.push_str("<|eot_id|>");
                    }
                    "assistant" => {
                        prompt.push_str(
                            "<|start_header_id|>assistant<|end_header_id|>\n\n",
                        );
                        prompt.push_str(&message.content);
                        prompt.push_str("<|eot_id|>");
                    }
                    _ => {
                        return Err(ProviderError::Other(format!(
                            "Unsupported role: {}",
                            message.role
                        )));
                    }
                }
            }

            prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");

            let encoding = tokenizer
                .encode(prompt.as_str(), false)
                .map_err(|e| ProviderError::Other(format!("Tokenization failed: {}", e)))?;
            let token_ids = encoding.get_ids().to_vec();

            Ok((prompt, token_ids))
        } else {
            let token_ids = self.encode_without_tokenizer(messages)?;
            Ok((String::new(), token_ids))
        }
    }

    fn encode_without_tokenizer(&self, messages: &[Message]) -> Result<Vec<u32>> {
        const BOS: u32 = 128_000;
        const START_HEADER: u32 = 128_006;
        const END_HEADER: u32 = 128_007;
        const EOT: u32 = 128_009;

        let mut token_ids = vec![BOS];

        for message in messages {
            token_ids.push(START_HEADER);

            let role_tokens = match message.role.as_str() {
                "system" => simple_encode("system"),
                "user" => simple_encode("user"),
                "assistant" => simple_encode("assistant"),
                _ => {
                    return Err(ProviderError::Other(format!(
                        "Unsupported role: {}",
                        message.role
                    )))
                }
            };
            token_ids.extend(role_tokens);
            token_ids.push(END_HEADER);
            token_ids.push(198);
            token_ids.push(198);

            if let Some(ref context) = message.context {
                token_ids.extend(simple_encode(context));
                token_ids.push(198);
                token_ids.push(198);
            }

            token_ids.extend(simple_encode(&message.content));
            token_ids.push(EOT);
        }

        token_ids.push(START_HEADER);
        token_ids.extend(simple_encode("assistant"));
        token_ids.push(END_HEADER);
        token_ids.push(198);
        token_ids.push(198);

        Ok(token_ids)
    }

    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let tokenizer = self
            ._tokenizer
            .as_ref()
            .ok_or_else(|| ProviderError::Other("Tokenizer not loaded".to_string()))?;

        let encoding = tokenizer
            .encode(prompt, false)
            .map_err(|e| ProviderError::Other(format!("Tokenization failed: {}", e)))?;

        let mut input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let mut generated_text = String::new();

        info!(
            "Starting generation with {} input tokens, max {} new tokens",
            input_ids.len(),
            max_tokens
        );

        for step in 0..max_tokens {
            let input_floats: Vec<f32> = input_ids.iter().map(|&id| id as f32).collect();

            let mut logits = vec![0.0f32; self._vocab_size];

            self.predict(&input_floats, &mut logits)?;

            let next_token_id = argmax(&logits);

            if next_token_id == 0 || next_token_id >= self._vocab_size as u32 {
                debug!("EOS or invalid token {} at step {}", next_token_id, step);
                break;
            }

            input_ids.push(next_token_id);

            let token_str = tokenizer
                .decode(&[next_token_id], true)
                .map_err(|e| ProviderError::Other(format!("Detokenization failed: {}", e)))?;

            generated_text.push_str(&token_str);

            if step % 10 == 0 {
                debug!("Generated {} tokens", step + 1);
            }
        }

        info!("Generation complete: {} total tokens", input_ids.len());
        Ok(generated_text)
    }

    fn predict_stateful(&self, input_ids: &[u32], output: &mut [f32]) -> Result<()> {
        let input_ids_i32: Vec<i32> = input_ids.iter().map(|&id| id as i32).collect();

        let seq_len = input_ids.len();
        let causal_mask = create_causal_mask(seq_len);

        let state_ptr = self.state.as_ref().map(|s| s.0).unwrap_or(std::ptr::null_mut());

        let result = unsafe {
            coreml_predict_stateful(
                self.model.0,
                state_ptr,
                input_ids_i32.as_ptr(),
                input_ids_i32.len(),
                causal_mask.as_ptr(),
                causal_mask.len(),
                output.as_mut_ptr(),
                output.len(),
            )
        };

        if result != 0 {
            return Err(ProviderError::Other(format!(
                "CoreML stateful prediction failed with code: {}",
                result
            )));
        }

        Ok(())
    }

    pub fn predict(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        let input_name = CString::new(self.input_name.as_str())
            .map_err(|e| ProviderError::Other(format!("Invalid input name: {}", e)))?;

        let output_name = CString::new(self.output_name.as_str())
            .map_err(|e| ProviderError::Other(format!("Invalid output name: {}", e)))?;

        let result = unsafe {
            coreml_predict_multiarray(
                self.model.0,
                input_name.as_ptr(),
                input.as_ptr(),
                input.len(),
                output_name.as_ptr(),
                output.as_mut_ptr(),
                output.len(),
            )
        };

        if result != 0 {
            return Err(ProviderError::Other(format!(
                "CoreML prediction failed with code: {}",
                result
            )));
        }

        Ok(())
    }

    pub fn get_input_shape(&self, max_dims: usize) -> Result<Vec<i64>> {
        let input_name = CString::new(self.input_name.as_str())
            .map_err(|e| ProviderError::Other(format!("Invalid input name: {}", e)))?;

        let mut shape = vec![0i64; max_dims];

        let dims = unsafe {
            coreml_get_input_shape(
                self.model.0,
                input_name.as_ptr(),
                shape.as_mut_ptr(),
                max_dims,
            )
        };

        if dims < 0 {
            return Err(ProviderError::Other(format!(
                "Failed to get input shape: {}",
                dims
            )));
        }

        shape.truncate(dims as usize);
        Ok(shape)
    }

    /// Reset sequence state for a new conversation (equivalent to passing nil state initially).
    pub fn reset_state(&mut self) {
        if let Some(CoreMLStateRef(state)) = self.state.take() {
            unsafe { coreml_free_state(state) };
        }

        let new_state = unsafe { coreml_make_state(self.model.0) };
        if !new_state.is_null() {
            self.state = Some(CoreMLStateRef(new_state));
        }
    }
}

impl Drop for CoreMLProvider {
    fn drop(&mut self) {
        // Free state first, then model.
        if let Some(CoreMLStateRef(state)) = self.state {
            unsafe { coreml_free_state(state) };
        }

        unsafe {
            coreml_free_model(self.model.0);
        }
        debug!("CoreML model freed: {}", self.model_path);
    }
}

#[async_trait]
impl Provider for CoreMLProvider {
    async fn chat<'a>(
        &'a self,
        request: ChatRequest,
        mut callback: Box<dyn FnMut(ChatResponse) + Send + 'a>,
    ) -> Result<()> {
        let (_prompt_text, mut input_ids) = self.format_chat_prompt(&request.messages)?;

        let max_tokens = 512;

        info!(
            "Starting chat generation with {} input tokens, max {} new tokens",
            input_ids.len(),
            max_tokens
        );

        // Note: state is stored on self and reused every call; reset_state can be used between conversations.
        for step in 0..max_tokens {
            let mut logits = vec![0.0f32; self._vocab_size];

            self.predict_stateful(&input_ids, &mut logits)?;

            let next_token_id = if request.temperature > 0.0 {
                sample_with_temperature(&logits, request.temperature)
            } else {
                argmax(&logits)
            };

            if next_token_id == 0 || next_token_id >= self._vocab_size as u32 {
                debug!("EOS or invalid token {} at step {}", next_token_id, step);

                callback(ChatResponse {
                    model: request.model.clone(),
                    content: String::new(),
                    done: true,
                    message: Message::assistant(None, ""),
                });
                break;
            }

            input_ids.push(next_token_id);

            let token_str = if let Some(ref tokenizer) = self._tokenizer {
                tokenizer
                    .decode(&[next_token_id], true)
                    .map_err(|e| ProviderError::Other(format!("Detokenization failed: {}", e)))?
            } else {
                simple_decode(&[next_token_id])
            };

            callback(ChatResponse {
                model: request.model.clone(),
                content: token_str.clone(),
                done: false,
                message: Message::assistant(None, token_str),
            });

            if step % 10 == 0 {
                debug!("Generated {} tokens", step + 1);
            }
        }

        info!("Chat generation complete: {} total tokens", input_ids.len());
        Ok(())
    }

    async fn embed(&self, _text: &str, _model: &EmbeddingModel) -> Result<Vec<f32>> {
        Err(ProviderError::Other(
            "CoreML provider does not support embed interface. Use predict() directly.".to_string(),
        ))
    }
}
