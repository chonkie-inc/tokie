use pyo3::prelude::*;
use pyo3::create_exception;
use pyo3::types::PyBytes;

create_exception!(tokie, TokieError, pyo3::exceptions::PyException);

fn to_py_err(e: impl std::fmt::Display) -> PyErr {
    TokieError::new_err(e.to_string())
}

/// Result of encoding a pair of texts (e.g. for cross-encoder models).
#[pyclass(name = "EncodingPair")]
struct PyEncodingPair {
    #[pyo3(get)]
    ids: Vec<u32>,
    attention_mask: Vec<u8>,
    type_ids: Vec<u8>,
}

#[pymethods]
impl PyEncodingPair {
    #[getter]
    fn attention_mask(&self) -> Vec<u32> {
        self.attention_mask.iter().map(|&x| x as u32).collect()
    }

    #[getter]
    fn type_ids(&self) -> Vec<u32> {
        self.type_ids.iter().map(|&x| x as u32).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "EncodingPair(ids=[...{}], attention_mask=[...{}], type_ids=[...{}])",
            self.ids.len(),
            self.attention_mask.len(),
            self.type_ids.len(),
        )
    }

    fn __len__(&self) -> usize {
        self.ids.len()
    }
}

/// Fast, correct tokenizer. Supports BPE, WordPiece, and Unigram.
#[pyclass(name = "Tokenizer")]
struct PyTokenizer {
    inner: tokie_core::Tokenizer,
}

#[pymethods]
impl PyTokenizer {
    /// Load a tokenizer from a HuggingFace tokenizer.json file.
    #[staticmethod]
    fn from_json(path: &str) -> PyResult<Self> {
        let inner = tokie_core::Tokenizer::from_json(path).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Load a tokenizer from a .tkz binary file.
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        let inner = tokie_core::Tokenizer::from_file(path).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Download and load a tokenizer from the HuggingFace Hub.
    /// Tries .tkz first, then falls back to tokenizer.json.
    #[staticmethod]
    fn from_pretrained(py: Python<'_>, repo_id: &str) -> PyResult<Self> {
        let repo = repo_id.to_string();
        let inner = py.allow_threads(|| {
            tokie_core::Tokenizer::from_pretrained(&repo).map_err(to_py_err)
        })?;
        Ok(Self { inner })
    }

    /// Encode text into token IDs.
    #[pyo3(signature = (text, add_special_tokens=true))]
    fn encode(&self, py: Python<'_>, text: &str, add_special_tokens: bool) -> Vec<u32> {
        let text = text.to_string();
        let inner = &self.inner;
        py.allow_threads(|| inner.encode(&text, add_special_tokens))
    }

    /// Encode a pair of texts (e.g. for cross-encoder models).
    #[pyo3(signature = (text_a, text_b, add_special_tokens=true))]
    fn encode_pair(
        &self,
        py: Python<'_>,
        text_a: &str,
        text_b: &str,
        add_special_tokens: bool,
    ) -> PyEncodingPair {
        let a = text_a.to_string();
        let b = text_b.to_string();
        let inner = &self.inner;
        let pair = py.allow_threads(|| inner.encode_pair(&a, &b, add_special_tokens));
        PyEncodingPair {
            ids: pair.ids,
            attention_mask: pair.attention_mask,
            type_ids: pair.type_ids,
        }
    }

    /// Encode raw bytes into token IDs.
    fn encode_bytes(&self, data: &[u8]) -> Vec<u32> {
        self.inner.encode_bytes(data)
    }

    /// Decode token IDs back to a string. Returns None if not valid UTF-8.
    fn decode(&self, tokens: Vec<u32>) -> Option<String> {
        self.inner.decode(&tokens)
    }

    /// Decode token IDs back to raw bytes.
    fn decode_bytes<'py>(&self, py: Python<'py>, tokens: Vec<u32>) -> Bound<'py, PyBytes> {
        let bytes = self.inner.decode_bytes(&tokens);
        PyBytes::new(py, &bytes)
    }

    /// Count the number of tokens in the text.
    fn count_tokens(&self, py: Python<'_>, text: &str) -> usize {
        let text = text.to_string();
        let inner = &self.inner;
        py.allow_threads(|| inner.count_tokens(&text))
    }

    /// Save the tokenizer to a .tkz binary file.
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.to_file(path).map_err(to_py_err)
    }

    /// The vocabulary size.
    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn __repr__(&self) -> String {
        format!("Tokenizer(vocab_size={})", self.inner.vocab_size())
    }
}

#[pymodule]
fn tokie(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    m.add_class::<PyEncodingPair>()?;
    m.add("TokieError", m.py().get_type::<TokieError>())?;
    Ok(())
}
