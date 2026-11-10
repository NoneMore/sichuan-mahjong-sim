//! Minimal deterministic NumPy `.npy` writer and strict reader.
//!
//! Preparation writes version 1.0 headers, little-endian scalar descriptors,
//! C order, and a header padded so the data offset is 64-byte aligned, matching
//! NumPy's conventional layout. The reader accepts only the exact descriptors
//! this module writes; cache format version 1 is therefore intentionally strict.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use sha2::{Digest, Sha256};

use super::error::{invalid_array, io, Result};

const MAGIC: [u8; 6] = *b"\x93NUMPY";
const PREFIX_LEN: usize = 10;
const CHUNK_VALUES: usize = 4096;

/// Scalar layout of one cache array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NpyDtype {
    /// Little-endian IEEE-754 binary32.
    Float32,
    /// Little-endian signed 64-bit integer.
    Int64,
    /// Little-endian unsigned 64-bit integer.
    Uint64,
    /// Single unsigned byte.
    Uint8,
}

impl NpyDtype {
    pub(super) const fn label(self) -> &'static str {
        match self {
            NpyDtype::Float32 => "float32",
            NpyDtype::Int64 => "int64",
            NpyDtype::Uint64 => "uint64",
            NpyDtype::Uint8 => "uint8",
        }
    }

    pub(super) const fn descr(self) -> &'static str {
        match self {
            NpyDtype::Float32 => "<f4",
            NpyDtype::Int64 => "<i8",
            NpyDtype::Uint64 => "<u8",
            NpyDtype::Uint8 => "|u1",
        }
    }

    pub(super) const fn width(self) -> usize {
        match self {
            NpyDtype::Float32 => 4,
            NpyDtype::Int64 | NpyDtype::Uint64 => 8,
            NpyDtype::Uint8 => 1,
        }
    }

    fn from_descr(descr: &str) -> Option<Self> {
        match descr {
            "<f4" => Some(NpyDtype::Float32),
            "<i8" => Some(NpyDtype::Int64),
            "<u8" => Some(NpyDtype::Uint64),
            "|u1" => Some(NpyDtype::Uint8),
            _ => None,
        }
    }

    pub(super) fn from_label(label: &str) -> Option<Self> {
        match label {
            "float32" => Some(NpyDtype::Float32),
            "int64" => Some(NpyDtype::Int64),
            "uint64" => Some(NpyDtype::Uint64),
            "uint8" => Some(NpyDtype::Uint8),
            _ => None,
        }
    }
}

/// Scalar values that can be stored in a version 1 cache array.
pub(super) trait NpyElement: Copy {
    const DTYPE: NpyDtype;

    fn append_le(self, out: &mut [u8]);

    fn read_le(bytes: &[u8]) -> Self;
}

impl NpyElement for f32 {
    const DTYPE: NpyDtype = NpyDtype::Float32;

    fn append_le(self, out: &mut [u8]) {
        out.copy_from_slice(&self.to_le_bytes());
    }

    fn read_le(bytes: &[u8]) -> Self {
        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }
}

impl NpyElement for i64 {
    const DTYPE: NpyDtype = NpyDtype::Int64;

    fn append_le(self, out: &mut [u8]) {
        out.copy_from_slice(&self.to_le_bytes());
    }

    fn read_le(bytes: &[u8]) -> Self {
        let mut array = [0u8; 8];
        array.copy_from_slice(bytes);
        i64::from_le_bytes(array)
    }
}

impl NpyElement for u64 {
    const DTYPE: NpyDtype = NpyDtype::Uint64;

    fn append_le(self, out: &mut [u8]) {
        out.copy_from_slice(&self.to_le_bytes());
    }

    fn read_le(bytes: &[u8]) -> Self {
        let mut array = [0u8; 8];
        array.copy_from_slice(bytes);
        u64::from_le_bytes(array)
    }
}

impl NpyElement for u8 {
    const DTYPE: NpyDtype = NpyDtype::Uint8;

    fn append_le(self, out: &mut [u8]) {
        out[0] = self;
    }

    fn read_le(bytes: &[u8]) -> Self {
        bytes[0]
    }
}

/// Writes one C-order little-endian `.npy` array, returning its SHA-256 and
/// exact byte length.
pub(super) fn write_array<T: NpyElement>(
    path: &Path,
    shape: &[u64],
    values: &[T],
) -> Result<(String, u64)> {
    let expected = element_count(path, shape)?;
    if expected != values.len() as u64 {
        return Err(invalid_array(
            path,
            format!(
                "shape {shape:?} requires {expected} values, got {}",
                values.len()
            ),
        ));
    }
    let file = File::create(path).map_err(|error| io(path, error))?;
    let mut writer = HashingWriter {
        inner: BufWriter::new(file),
        hasher: Sha256::new(),
        bytes: 0,
    };
    writer
        .write_all(&header_bytes(T::DTYPE, shape))
        .map_err(|error| io(path, error))?;
    let mut scratch = vec![0u8; CHUNK_VALUES * T::DTYPE.width()];
    for chunk in values.chunks(CHUNK_VALUES) {
        let width = T::DTYPE.width();
        for (index, value) in chunk.iter().enumerate() {
            value.append_le(&mut scratch[index * width..(index + 1) * width]);
        }
        writer
            .write_all(&scratch[..chunk.len() * width])
            .map_err(|error| io(path, error))?;
    }
    let (inner, sha256, bytes) = writer.finish().map_err(|error| io(path, error))?;
    inner
        .get_ref()
        .sync_all()
        .map_err(|error| io(path, error))?;
    drop(inner);
    Ok((sha256, bytes))
}

/// Validates an array file's header, dtype, shape, and exact file length.
/// Returns its SHA-256 and byte length.
pub(super) fn verify_array(path: &Path, dtype: NpyDtype, shape: &[u64]) -> Result<(String, u64)> {
    let (header, _reader) = read_header(path)?;
    if header.dtype != dtype {
        return Err(invalid_array(
            path,
            format!(
                "dtype is {}, expected {}",
                header.dtype.label(),
                dtype.label()
            ),
        ));
    }
    if header.shape != shape {
        return Err(invalid_array(
            path,
            format!("shape is {:?}, expected {shape:?}", header.shape),
        ));
    }
    if header.fortran_order {
        return Err(invalid_array(path, "Fortran order is not supported"));
    }
    let mut file = File::open(path).map_err(|error| io(path, error))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut bytes = 0u64;
    loop {
        let read = file.read(&mut buffer).map_err(|error| io(path, error))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        bytes += read as u64;
    }
    Ok((lower_hex(&hasher.finalize()), bytes))
}

/// Reads a complete array of `T`, checking its declared dtype and shape.
pub(super) fn read_array<T: NpyElement>(path: &Path, shape: &[u64]) -> Result<Vec<T>> {
    let (header, mut reader) = read_header(path)?;
    if header.dtype != T::DTYPE {
        return Err(invalid_array(
            path,
            format!(
                "dtype is {}, expected {}",
                header.dtype.label(),
                T::DTYPE.label()
            ),
        ));
    }
    if header.shape != shape {
        return Err(invalid_array(
            path,
            format!("shape is {:?}, expected {shape:?}", header.shape),
        ));
    }
    let count = element_count(path, shape)?;
    let count = usize::try_from(count)
        .map_err(|_| invalid_array(path, "array element count does not fit usize"))?;
    let width = T::DTYPE.width();
    let byte_count = count
        .checked_mul(width)
        .ok_or_else(|| invalid_array(path, "array byte length overflows usize"))?;
    let mut bytes = vec![0u8; byte_count];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| io(path, error))?;
    let mut values = Vec::with_capacity(count);
    for chunk in bytes.chunks_exact(width) {
        values.push(T::read_le(chunk));
    }
    Ok(values)
}

fn element_count(path: &Path, shape: &[u64]) -> Result<u64> {
    if shape.is_empty() {
        return Err(invalid_array(
            path,
            "zero-dimensional arrays are not supported",
        ));
    }
    let mut count = 1u64;
    for &dimension in shape {
        count = count
            .checked_mul(dimension)
            .ok_or_else(|| invalid_array(path, "shape element count overflows u64"))?;
    }
    Ok(count)
}

fn header_bytes(dtype: NpyDtype, shape: &[u64]) -> Vec<u8> {
    let shape_text = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let inner = shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        format!("({inner})")
    };
    let dict = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        dtype.descr(),
        shape_text
    );
    let unpadded = PREFIX_LEN + dict.len() + 1;
    let padding = (64 - (unpadded % 64)) % 64;
    let header_len = dict.len() + padding + 1;
    let mut out = Vec::with_capacity(PREFIX_LEN + header_len);
    out.extend_from_slice(&MAGIC);
    out.push(1);
    out.push(0);
    out.extend_from_slice(&(header_len as u16).to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out.extend(std::iter::repeat(b' ').take(padding));
    out.push(b'\n');
    out
}

struct NpyHeader {
    dtype: NpyDtype,
    fortran_order: bool,
    shape: Vec<u64>,
}

fn read_header(path: &Path) -> Result<(NpyHeader, BufReader<File>)> {
    let file = File::open(path).map_err(|error| io(path, error))?;
    let file_len = file.metadata().map_err(|error| io(path, error))?.len();
    let mut reader = BufReader::new(file);
    let mut prefix = [0u8; PREFIX_LEN];
    reader
        .read_exact(&mut prefix)
        .map_err(|error| io(path, error))?;
    if prefix[..6] != MAGIC {
        return Err(invalid_array(path, "missing NPY magic prefix"));
    }
    if prefix[6] != 1 || prefix[7] != 0 {
        return Err(invalid_array(
            path,
            format!("unsupported NPY version {}.{}", prefix[6], prefix[7]),
        ));
    }
    let header_len = usize::from(u16::from_le_bytes([prefix[8], prefix[9]]));
    let mut header = vec![0u8; header_len];
    reader
        .read_exact(&mut header)
        .map_err(|error| io(path, error))?;
    let shape_text = match std::str::from_utf8(&header) {
        Ok(text) => text,
        Err(_) => return Err(invalid_array(path, "NPY header is not ASCII")),
    };
    let shape_text = shape_text.trim_end_matches(['\n', ' ']);
    if !shape_text.starts_with('{') || !shape_text.ends_with('}') {
        return Err(invalid_array(path, "malformed NPY header dictionary"));
    }

    let descr = extract_quoted(path, shape_text, "'descr': '")?;
    let dtype = NpyDtype::from_descr(descr)
        .ok_or_else(|| invalid_array(path, format!("unsupported NPY dtype {descr:?}")))?;
    let fortran_order = match extract_value(path, shape_text, "'fortran_order': ")? {
        "True" => true,
        "False" => false,
        value => {
            return Err(invalid_array(
                path,
                format!("invalid fortran_order {value:?}"),
            ))
        }
    };
    let shape_text = extract_paren_value(path, shape_text, "'shape': (")?;
    let mut shape = Vec::new();
    for part in shape_text.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let dimension = part
            .parse::<u64>()
            .map_err(|_| invalid_array(path, format!("invalid shape dimension {part:?}")))?;
        shape.push(dimension);
    }
    if shape.is_empty() {
        return Err(invalid_array(
            path,
            "zero-dimensional arrays are not supported",
        ));
    }
    let expected_data = element_count(path, &shape)?
        .checked_mul(dtype.width() as u64)
        .ok_or_else(|| invalid_array(path, "array byte length overflows u64"))?;
    let expected_len = u64::try_from(PREFIX_LEN + header_len)
        .ok()
        .and_then(|offset| offset.checked_add(expected_data))
        .ok_or_else(|| invalid_array(path, "array byte length overflows u64"))?;
    if file_len != expected_len {
        return Err(invalid_array(
            path,
            format!("file length {file_len} does not match expected {expected_len}"),
        ));
    }
    Ok((
        NpyHeader {
            dtype,
            fortran_order,
            shape,
        },
        reader,
    ))
}

fn extract_quoted<'a>(path: &Path, text: &'a str, key: &str) -> Result<&'a str> {
    let start = text
        .find(key)
        .ok_or_else(|| invalid_array(path, format!("missing {key:?} in NPY header")))?
        + key.len();
    let rest = &text[start..];
    let end = rest
        .find('\'')
        .ok_or_else(|| invalid_array(path, format!("unterminated {key:?} in NPY header")))?;
    Ok(&rest[..end])
}

fn extract_value<'a>(path: &Path, text: &'a str, key: &str) -> Result<&'a str> {
    let start = text
        .find(key)
        .ok_or_else(|| invalid_array(path, format!("missing {key:?} in NPY header")))?
        + key.len();
    let rest = &text[start..];
    let end = rest
        .find(',')
        .or_else(|| rest.find('}'))
        .ok_or_else(|| invalid_array(path, format!("unterminated {key:?} in NPY header")))?;
    Ok(rest[..end].trim())
}

fn extract_paren_value<'a>(path: &Path, text: &'a str, key: &str) -> Result<&'a str> {
    let start = text
        .find(key)
        .ok_or_else(|| invalid_array(path, format!("missing {key:?} in NPY header")))?
        + key.len();
    let rest = &text[start..];
    let end = rest
        .find(')')
        .ok_or_else(|| invalid_array(path, format!("unterminated {key:?} in NPY header")))?;
    Ok(&rest[..end])
}

struct HashingWriter<W> {
    inner: W,
    hasher: Sha256,
    bytes: u64,
}

impl<W: Write> HashingWriter<W> {
    fn finish(mut self) -> std::io::Result<(W, String, u64)> {
        self.flush()?;
        let digest = self.hasher.finalize();
        let bytes = self.bytes;
        let Self {
            inner,
            hasher: _,
            bytes: _,
        } = self;
        Ok((inner, lower_hex(&digest), bytes))
    }
}

impl<W: Write> Write for HashingWriter<W> {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        let written = self.inner.write(buffer)?;
        self.hasher.update(&buffer[..written]);
        self.bytes += written as u64;
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

fn lower_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(char::from_digit(u32::from(byte >> 4), 16).expect("hex digit"));
        out.push(char::from_digit(u32::from(byte & 0xf), 16).expect("hex digit"));
    }
    out
}
