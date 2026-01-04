//! CoreML provider implementation for macOS.
//!
//! This module provides inference using Apple's CoreML framework.
//! Only available on macOS with the `coreml` feature enabled.

use crate::provider::{ChatRequest, ChatResponse, Provider, ProviderError, Result};
use crate::models::EmbeddingModel;
use crate::Config;
use async_trait::async_trait;
use nucleus_plugin::PluginRegistry;
use tokenizers::Tokenizer;
use std::ffi::{CString, c_char, c_float, c_int, c_void};
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info, warn};

#[repr(C)]
struct CoreMLModelRef(*mut c_void);

unsafe impl Send for CoreMLModelRef {}
unsafe impl Sync for CoreMLModelRef {}

extern "C" {
    fn coreml_load_model(model_path: *const c_char) -> *mut c_void;
    
    fn coreml_predict_multiarray(
        model: *mut c_void,
        input_name: *const c_char,
        input_data: *const c_float,
        input_size: usize,
        output_name: *const c_char,
        output_data: *mut c_float,
        output_size: usize,
    ) -> c_int;
    
    fn coreml_free_model(model: *mut c_void);
    
    fn coreml_get_input_shape(
        model: *mut c_void,
        input_name: *const c_char,
        shape_out: *mut i64,
        max_dims: usize,
    ) -> c_int;
}

fn argmax(logits: &[f32]) -> u32 {
    logits.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

pub struct CoreMLProvider {
    model: CoreMLModelRef,
    model_path: String,
    input_name: String,
    output_name: String,
    _registry: Arc<PluginRegistry>,
    _config: Config,
    _tokenizer: Option<Tokenizer>,
    _vocab_size: usize,
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
            return Err(ProviderError::Other(
                format!("CoreML model not found: {}", path.display())
            ));
        }
        
        let path_str = path.to_str()
            .ok_or_else(|| ProviderError::Other("Invalid UTF-8 in model path".to_string()))?;
        
        let c_path = CString::new(path_str)
            .map_err(|e| ProviderError::Other(format!("Invalid path: {}", e)))?;
        
        let handle = unsafe { coreml_load_model(c_path.as_ptr()) };
        
        if handle.is_null() {
            return Err(ProviderError::Other(
                "Failed to load CoreML model".to_string()
            ));
        }

        let tokenizer_parent = Path::new(&expanded_path).parent()
            .ok_or_else(|| ProviderError::Other("Invalid model path".to_string()))?;
        let tokenizer_path = tokenizer_parent.join("tokenizer.json");
        
        let tokenizer = if tokenizer_path.exists() {
            let tok = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| ProviderError::Other(format!("Tokenizer load failed: {}", e)))?;
            
            let vocab_size = tok.get_vocab_size(false);
            info!("Tokenizer loaded: {} tokens", vocab_size);
            
            Some(tok)
        } else {
            info!("No tokenizer.json found at {:?}, using default vocab size from config", tokenizer_path);
            None
        };
        
        let vocab_size = tokenizer.as_ref()
            .map(|t| t.get_vocab_size(false))
            .unwrap_or_else(|| {
                debug!("Using context_length 128256 as default vocab size");
                128256
            });
        
        info!("CoreML model loaded: {}", path.display());
        
        Ok(Arc::new(Self {
            model: CoreMLModelRef(handle),
            model_path: path_str.to_string(),
            input_name: "inputIds".to_string(),
            output_name: "logits".to_string(),
            _registry: registry,
            _config: config.clone(),
            _tokenizer: tokenizer,
            _vocab_size: vocab_size,
        }))
    }

    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let tokenizer = self._tokenizer.as_ref()
            .ok_or_else(|| ProviderError::Other("Tokenizer not loaded".to_string()))?;
        
        let encoding = tokenizer.encode(prompt, false)
            .map_err(|e| ProviderError::Other(format!("Tokenization failed: {}", e)))?;
        
        let mut input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let mut generated_text = String::new();
        
        info!("Starting generation with {} input tokens, max {} new tokens", input_ids.len(), max_tokens);
        
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
            
            let token_str = tokenizer.decode(&[next_token_id], true)
                .map_err(|e| ProviderError::Other(format!("Detokenization failed: {}", e)))?;
            
            generated_text.push_str(&token_str);
            
            if step % 10 == 0 {
                debug!("Generated {} tokens", step + 1);
            }
        }
        
        info!("Generation complete: {} total tokens", input_ids.len());
        Ok(generated_text)
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
            return Err(ProviderError::Other(
                format!("CoreML prediction failed with code: {}", result)
            ));
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
            return Err(ProviderError::Other(
                format!("Failed to get input shape: {}", dims)
            ));
        }
        
        shape.truncate(dims as usize);
        Ok(shape)
    }
}

impl Drop for CoreMLProvider {
    fn drop(&mut self) {
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
        _request: ChatRequest,
        _callback: Box<dyn FnMut(ChatResponse) + Send + 'a>,
    ) -> Result<()> {
        Err(ProviderError::Other(
            "CoreML provider does not support chat interface. Use predict() directly.".to_string()
        ))
    }
    
    async fn embed(&self, _text: &str, _model: &EmbeddingModel) -> Result<Vec<f32>> {
        Err(ProviderError::Other(
            "CoreML provider does not support embed interface. Use predict() directly.".to_string()
        ))
    }
}

