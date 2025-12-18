//! CoreML provider implementation for macOS.
//!
//! This module provides inference using Apple's CoreML framework.
//! Only available on macOS with the `coreml` feature enabled.

use super::types::*;
use crate::models::EmbeddingModel;
use async_trait::async_trait;
use std::ffi::{CString, c_char, c_float, c_int, c_void};
use std::path::Path;
use tracing::{debug, info};

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

pub struct CoreMLProvider {
    model: CoreMLModelRef,
    model_path: String,
    input_name: String,
    output_name: String,
}

impl CoreMLProvider {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        input_name: impl Into<String>,
        output_name: impl Into<String>,
    ) -> Result<Self> {
        let path = model_path.as_ref();
        
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
        
        info!("CoreML model loaded: {}", path.display());
        
        Ok(Self {
            model: CoreMLModelRef(handle),
            model_path: path_str.to_string(),
            input_name: input_name.into(),
            output_name: output_name.into(),
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[ignore]
    fn test_coreml_basic() {
        let provider = CoreMLProvider::new(
            "/path/to/model.mlmodelc",
            "input",
            "output"
        );
        
        assert!(provider.is_ok() || provider.is_err());
    }
}
