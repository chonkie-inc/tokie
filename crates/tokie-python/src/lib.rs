use std::sync::RwLock;

use pyo3::prelude::*;
use pyo3::create_exception;
use pyo3::types::PyBytes;

create_exception!(tokie, TokieError, pyo3::exceptions::PyException);

fn to_py_err(e: impl std::fmt::Display) -> PyErr {
    TokieError::new_err(e.to_string())
}

/// Result of encoding text, with token IDs, attention mask, and type IDs.
#[pyclass(name = "Encoding")]
#[derive(Clone)]
struct PyEncoding {
    #[pyo3(get)]
    ids: Vec<u32>,
    attention_mask_inner: Vec<u8>,
    type_ids_inner: Vec<u8>,
}

#[pymethods]
impl PyEncoding {
    #[getter]
    fn attention_mask(&self) -> Vec<u32> {
        self.attention_mask_inner.iter().map(|&x| x as u32).collect()
    }

    #[getter]
    fn type_ids(&self) -> Vec<u32> {
        self.type_ids_inner.iter().map(|&x| x as u32).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Encoding(ids=[...{}], attention_mask=[...{}], type_ids=[...{}])",
            self.ids.len(),
            self.attention_mask_inner.len(),
            self.type_ids_inner.len(),
        )
    }

    fn __len__(&self) -> usize {
        self.ids.len()
    }
}

impl From<tokie_core::Encoding> for PyEncoding {
    fn from(enc: tokie_core::Encoding) -> Self {
        Self {
            ids: enc.ids,
            attention_mask_inner: enc.attention_mask,
            type_ids_inner: enc.type_ids,
        }
    }
}

/// Fast, correct tokenizer. Supports BPE, WordPiece, and Unigram.
#[pyclass(name = "Tokenizer")]
struct PyTokenizer {
    inner: RwLock<tokie_core::Tokenizer>,
}

impl PyTokenizer {
    fn read(&self) -> std::sync::RwLockReadGuard<'_, tokie_core::Tokenizer> {
        self.inner.read().unwrap()
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, tokie_core::Tokenizer> {
        self.inner.write().unwrap()
    }
}

#[pymethods]
impl PyTokenizer {
    /// Load a tokenizer from a HuggingFace tokenizer.json file.
    #[staticmethod]
    fn from_json(path: &str) -> PyResult<Self> {
        let inner = tokie_core::Tokenizer::from_json(path).map_err(to_py_err)?;
        Ok(Self { inner: RwLock::new(inner) })
    }

    /// Load a tokenizer from a .tkz binary file.
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        let inner = tokie_core::Tokenizer::from_file(path).map_err(to_py_err)?;
        Ok(Self { inner: RwLock::new(inner) })
    }

    /// Download and load a tokenizer from the HuggingFace Hub.
    /// Tries .tkz first, then falls back to tokenizer.json.
    #[staticmethod]
    fn from_pretrained(py: Python<'_>, repo_id: &str) -> PyResult<Self> {
        let repo = repo_id.to_string();
        let inner = py.allow_threads(|| {
            tokie_core::Tokenizer::from_pretrained(&repo).map_err(to_py_err)
        })?;
        Ok(Self { inner: RwLock::new(inner) })
    }

    /// Encode text into an Encoding (ids, attention_mask, type_ids).
    #[pyo3(signature = (text, add_special_tokens=true))]
    fn encode(&self, py: Python<'_>, text: &str, add_special_tokens: bool) -> PyEncoding {
        let text = text.to_string();
        let inner = self.read();
        py.allow_threads(|| inner.encode(&text, add_special_tokens)).into()
    }

    /// Encode a pair of texts (e.g. for cross-encoder models).
    #[pyo3(signature = (text_a, text_b, add_special_tokens=true))]
    fn encode_pair(
        &self,
        py: Python<'_>,
        text_a: &str,
        text_b: &str,
        add_special_tokens: bool,
    ) -> PyEncoding {
        let a = text_a.to_string();
        let b = text_b.to_string();
        let inner = self.read();
        py.allow_threads(|| inner.encode_pair(&a, &b, add_special_tokens)).into()
    }

    /// Encode raw bytes into token IDs.
    fn encode_bytes(&self, data: &[u8]) -> Vec<u32> {
        self.read().encode_bytes(data)
    }

    /// Decode token IDs back to a string. Returns None if not valid UTF-8.
    fn decode(&self, tokens: Vec<u32>) -> Option<String> {
        self.read().decode(&tokens)
    }

    /// Decode token IDs back to raw bytes.
    fn decode_bytes<'py>(&self, py: Python<'py>, tokens: Vec<u32>) -> Bound<'py, PyBytes> {
        let bytes = self.read().decode_bytes(&tokens);
        PyBytes::new(py, &bytes)
    }

    /// Encode multiple texts in parallel, returning a list of Encoding objects.
    #[pyo3(signature = (texts, add_special_tokens=true))]
    fn encode_batch(&self, py: Python<'_>, texts: Vec<String>, add_special_tokens: bool) -> Vec<PyEncoding> {
        let inner = self.read();
        py.allow_threads(|| {
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            inner.encode_batch(&text_refs, add_special_tokens)
                .into_iter()
                .map(PyEncoding::from)
                .collect()
        })
    }

    /// Count the number of tokens in the text.
    fn count_tokens(&self, py: Python<'_>, text: &str) -> usize {
        let text = text.to_string();
        let inner = self.read();
        py.allow_threads(|| inner.count_tokens(&text))
    }

    /// Count tokens for multiple texts in parallel.
    fn count_tokens_batch(&self, py: Python<'_>, texts: Vec<String>) -> Vec<usize> {
        let inner = self.read();
        py.allow_threads(|| {
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            inner.count_tokens_batch(&text_refs)
        })
    }

    /// Save the tokenizer to a .tkz binary file.
    fn save(&self, path: &str) -> PyResult<()> {
        self.read().to_file(path).map_err(to_py_err)
    }

    /// The vocabulary size.
    #[getter]
    fn vocab_size(&self) -> usize {
        self.read().vocab_size()
    }

    /// The pad token ID, if set.
    #[getter]
    fn pad_token_id(&self) -> Option<u32> {
        self.read().pad_token_id()
    }

    /// Enable padding for encode_batch.
    #[pyo3(signature = (*, direction="right", pad_id=0, pad_type_id=0, length=None, pad_to_multiple_of=None))]
    fn enable_padding(
        &self,
        direction: &str,
        pad_id: u32,
        pad_type_id: u8,
        length: Option<usize>,
        pad_to_multiple_of: Option<usize>,
    ) -> PyResult<()> {
        let direction = match direction {
            "right" => tokie_core::PaddingDirection::Right,
            "left" => tokie_core::PaddingDirection::Left,
            _ => return Err(TokieError::new_err("direction must be 'left' or 'right'")),
        };
        let strategy = match length {
            Some(n) => tokie_core::PaddingStrategy::Fixed(n),
            None => tokie_core::PaddingStrategy::BatchLongest,
        };
        let params = tokie_core::PaddingParams {
            strategy,
            direction,
            pad_to_multiple_of,
            pad_id,
            pad_type_id,
        };
        self.write().enable_padding(params);
        Ok(())
    }

    /// Enable truncation.
    #[pyo3(signature = (max_length, *, stride=0, strategy="longest_first", direction="right"))]
    fn enable_truncation(
        &self,
        max_length: usize,
        stride: usize,
        strategy: &str,
        direction: &str,
    ) -> PyResult<()> {
        let strategy = match strategy {
            "longest_first" => tokie_core::TruncationStrategy::LongestFirst,
            "only_first" => tokie_core::TruncationStrategy::OnlyFirst,
            "only_second" => tokie_core::TruncationStrategy::OnlySecond,
            _ => return Err(TokieError::new_err("strategy must be 'longest_first', 'only_first', or 'only_second'")),
        };
        let direction = match direction {
            "right" => tokie_core::TruncationDirection::Right,
            "left" => tokie_core::TruncationDirection::Left,
            _ => return Err(TokieError::new_err("direction must be 'left' or 'right'")),
        };
        let params = tokie_core::TruncationParams {
            max_length,
            strategy,
            direction,
            stride,
        };
        self.write().enable_truncation(params);
        Ok(())
    }

    /// Disable padding.
    fn no_padding(&self) {
        self.write().no_padding();
    }

    /// Disable truncation.
    fn no_truncation(&self) {
        self.write().no_truncation();
    }

    fn __repr__(&self) -> String {
        format!("Tokenizer(vocab_size={})", self.read().vocab_size())
    }
}

#[pymodule]
fn tokie(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    m.add_class::<PyEncoding>()?;
    m.add("TokieError", m.py().get_type::<TokieError>())?;
    Ok(())
}
