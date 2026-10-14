// SPDX-License-Identifier: AGPL-3.0-or-later
//! Thin CPython FFI. Heretic algorithms live in Rust; imported modules are third-party dependencies.
use anyhow::{bail, Result};
use pyo3::{
    prelude::*,
    types::{PyBool, PyDict, PyFloat, PyInt, PyList, PySlice, PyString, PyTuple},
};
use serde::Serialize;
use serde_json::Value;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static JOURNAL_DEPTH: AtomicUsize = AtomicUsize::new(0);
static DEFERRED_INTERRUPT: AtomicBool = AtomicBool::new(false);

/// Let Optuna finish a journal transaction before raising an asynchronous exception.
#[pyfunction]
pub fn handle_sigint(_signal: i32, _frame: &Bound<'_, PyAny>) -> PyResult<()> {
    if JOURNAL_DEPTH.load(Ordering::SeqCst) > 0 {
        DEFERRED_INTERRUPT.store(true, Ordering::SeqCst);
        Ok(())
    } else {
        Err(pyo3::exceptions::PyKeyboardInterrupt::new_err(()))
    }
}
struct JournalGuard(bool);
impl JournalGuard {
    fn new(value: &Bound<'_, PyAny>, callable: bool) -> Self {
        let module = if callable {
            value.getattr("__module__")
        } else {
            value.get_type().getattr("__module__")
        };
        let guarded = module
            .and_then(|x| x.extract::<String>())
            .is_ok_and(|x| x.starts_with("optuna"));
        if guarded {
            JOURNAL_DEPTH.fetch_add(1, Ordering::SeqCst);
        }
        Self(guarded)
    }
}
impl Drop for JournalGuard {
    fn drop(&mut self) {
        if self.0 {
            JOURNAL_DEPTH.fetch_sub(1, Ordering::SeqCst);
        }
    }
}
pub fn deliver_interrupt() -> PyResult<()> {
    if JOURNAL_DEPTH.load(Ordering::SeqCst) == 0 && DEFERRED_INTERRUPT.swap(false, Ordering::SeqCst)
    {
        Err(pyo3::exceptions::PyKeyboardInterrupt::new_err(()))
    } else {
        Ok(())
    }
}

#[derive(Clone)]
pub struct O<'py>(pub Bound<'py, PyAny>);
impl<'py> O<'py> {
    pub fn a(&self, path: &str) -> Result<Self> {
        let mut x = self.clone();
        for name in path.split('.') {
            let guard = JournalGuard::new(&x.0, false);
            let result = x.0.getattr(name);
            drop(guard);
            deliver_interrupt()?;
            x = Self(result?);
        }
        Ok(x)
    }
    pub fn set(&self, k: &str, v: &Self) -> Result<()> {
        self.0.setattr(k, &v.0)?;
        Ok(())
    }
    pub fn m(&self, name: &str, args: &[Self], kw: &[(&str, Self)]) -> Result<Self> {
        self.a(name)?.c(args, kw)
    }
    pub fn c(&self, args: &[Self], kw: &[(&str, Self)]) -> Result<Self> {
        let py = self.0.py();
        let d = PyDict::new(py);
        for (k, v) in kw {
            d.set_item(k, &v.0)?;
        }
        let tuple = PyTuple::new(py, args.iter().map(|x| &x.0))?;
        let guard = JournalGuard::new(&self.0, true);
        // The caller must receive a newly persisted trial before an interrupt
        // can be raised, so it can close that trial during recovery.
        let creates_trial = guard.0
            && self
                .0
                .getattr("__name__")
                .and_then(|v| v.extract::<String>())
                .is_ok_and(|v| v == "ask");
        let result = self.0.call(tuple, Some(&d));
        drop(guard);
        if !creates_trial {
            deliver_interrupt()?;
        }
        Ok(Self(result?))
    }
    pub fn idx(&self, key: &Self) -> Result<Self> {
        Ok(Self(self.0.get_item(&key.0)?))
    }
    pub fn item(&self, i: isize) -> Result<Self> {
        self.idx(&j(self.0.py(), i)?)
    }
    pub fn key(&self, k: &str) -> Result<Self> {
        self.idx(&j(self.0.py(), k)?)
    }
    pub fn list(&self) -> Result<Vec<Self>> {
        self.0.try_iter()?.map(|v| Ok(Self(v?))).collect()
    }
    pub fn len(&self) -> Result<usize> {
        Ok(self.0.len()?)
    }
    pub fn empty(&self) -> Result<bool> {
        Ok(!self.0.is_truthy()?)
    }
    pub fn is_none(&self) -> bool {
        self.0.is_none()
    }
    pub fn str(&self) -> Result<String> {
        Ok(self.0.str()?.to_string())
    }
    pub fn f64(&self) -> Result<f64> {
        Ok(self.0.extract()?)
    }
    pub fn usize(&self) -> Result<usize> {
        Ok(self.0.extract()?)
    }
    pub fn strings(&self) -> Result<Vec<String>> {
        Ok(self.0.extract()?)
    }
    pub fn json(&self) -> Result<Value> {
        from_python(&self.0)
    }
    pub fn add(&self, r: &Self) -> Result<Self> {
        self.m("__add__", &[r.clone()], &[])
    }
    pub fn sub(&self, r: &Self) -> Result<Self> {
        self.m("__sub__", &[r.clone()], &[])
    }
    pub fn mul(&self, r: &Self) -> Result<Self> {
        self.m("__mul__", &[r.clone()], &[])
    }
    pub fn div(&self, r: &Self) -> Result<Self> {
        self.m("__truediv__", &[r.clone()], &[])
    }
    pub fn mat(&self, r: &Self) -> Result<Self> {
        self.m("__matmul__", &[r.clone()], &[])
    }
    pub fn neg(&self) -> Result<Self> {
        self.m("__neg__", &[], &[])
    }
    pub fn dim(&self, i: usize) -> Result<usize> {
        self.a("shape")?.item(i as isize)?.usize()
    }
    pub fn instance_of(&self, class: &Self) -> Result<bool> {
        Ok(self.0.is_instance(&class.0)?)
    }
    pub fn tensor_slice(&self, parts: &[Self]) -> Result<Self> {
        self.idx(&tuple(self.0.py(), parts)?)
    }
}
pub fn module<'py>(py: Python<'py>, name: &str) -> Result<O<'py>> {
    Ok(O(py.import(name)?.into_any()))
}
pub fn dict(py: Python<'_>) -> O<'_> {
    O(PyDict::new(py).into_any())
}
pub fn configure_runtime(py: Python<'_>) -> Result<()> {
    // VIRTUAL_ENV alone does not affect an embedded interpreter's sys.path.
    // Select packages for this interpreter's ABI instead of the first lib/python* directory.
    if std::env::var("PYTHONPATH").is_ok_and(|s| !s.is_empty()) {
        return Ok(());
    }
    let sys = module(py, "sys")?;
    let major = sys.a("version_info.major")?.usize()?;
    let minor = sys.a("version_info.minor")?.usize()?;
    let mut candidates = Vec::new();
    if let Some(path) = std::env::var_os("VIRTUAL_ENV") {
        candidates.push(std::path::PathBuf::from(path));
    }
    candidates.push(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".venv"));
    if let Ok(executable) = std::env::current_exe() {
        candidates.extend(
            executable
                .ancestors()
                .skip(1)
                .take(4)
                .map(|p| p.join(".venv")),
        );
    }
    for environment in candidates {
        let unix = environment.join(format!("lib/python{major}.{minor}/site-packages"));
        let windows = environment.join("Lib/site-packages");
        let site = if unix.is_dir() {
            unix
        } else if cfg!(windows) && windows.is_dir() {
            windows
        } else {
            continue;
        };
        sys.a("path")?
            .m("insert", &[j(py, 0)?, j(py, site.to_string_lossy())?], &[])?;
        sys.set("prefix", &j(py, environment.to_string_lossy())?)?;
        sys.set("exec_prefix", &j(py, environment.to_string_lossy())?)?;
        break;
    }
    Ok(())
}
pub fn j<'py>(py: Python<'py>, v: impl Serialize) -> Result<O<'py>> {
    to_python(py, serde_json::to_value(v)?)
}
fn to_python(py: Python<'_>, value: Value) -> Result<O<'_>> {
    Ok(O(match value {
        Value::Null => py.None().into_bound(py),
        Value::Bool(v) => v.into_pyobject(py)?.to_owned().into_any(),
        Value::Number(v) => {
            if let Some(n) = v.as_i64() {
                n.into_pyobject(py)?.into_any()
            } else if let Some(n) = v.as_u64() {
                n.into_pyobject(py)?.into_any()
            } else {
                v.as_f64().unwrap().into_pyobject(py)?.into_any()
            }
        }
        Value::String(v) => PyString::new(py, &v).into_any(),
        Value::Array(v) => {
            let items = v
                .into_iter()
                .map(|v| to_python(py, v).map(|x| x.0))
                .collect::<Result<Vec<_>>>()?;
            PyList::new(py, items)?.into_any()
        }
        Value::Object(v) => {
            let dict = PyDict::new(py);
            for (k, v) in v {
                dict.set_item(k, to_python(py, v)?.0)?;
            }
            dict.into_any()
        }
    }))
}
fn from_python(v: &Bound<'_, PyAny>) -> Result<Value> {
    if v.is_none() {
        return Ok(Value::Null);
    }
    if v.is_instance_of::<PyBool>() {
        return Ok(Value::Bool(v.extract()?));
    }
    if v.is_instance_of::<PyInt>() {
        return Ok(if let Ok(n) = v.extract::<i64>() {
            Value::from(n)
        } else {
            Value::from(v.extract::<u64>()?)
        });
    }
    if v.is_instance_of::<PyString>() {
        return Ok(Value::String(v.extract()?));
    }
    if let Ok(dict) = v.downcast::<PyDict>() {
        let mut object = serde_json::Map::new();
        for (k, v) in dict.iter() {
            let key = if k.is_instance_of::<PyString>() {
                k.extract()?
            } else {
                k.str()?.to_string()
            };
            object.insert(key, from_python(&v)?);
        }
        return Ok(Value::Object(object));
    }
    if v.is_instance_of::<PyList>() || v.is_instance_of::<PyTuple>() {
        return Ok(Value::Array(
            v.try_iter()?
                .map(|v| from_python(&v?))
                .collect::<Result<_>>()?,
        ));
    }
    if v.is_instance_of::<PyFloat>() || v.extract::<f64>().is_ok() {
        return Ok(serde_json::to_value(v.extract::<f64>()?)?);
    }
    bail!(
        "Unsupported Python value for JSON: {}",
        v.get_type().name()?
    )
}
pub fn tuple<'py>(py: Python<'py>, v: &[O<'py>]) -> Result<O<'py>> {
    Ok(O(PyTuple::new(py, v.iter().map(|x| &x.0))?.into_any()))
}
pub fn list<'py>(py: Python<'py>, v: &[O<'py>]) -> Result<O<'py>> {
    Ok(O(PyList::new(py, v.iter().map(|x| &x.0))?.into_any()))
}
pub fn slice<'py>(py: Python<'py>, start: Option<isize>, stop: Option<isize>) -> Result<O<'py>> {
    Ok(O(PySlice::new(
        py,
        start.unwrap_or(0),
        stop.unwrap_or(isize::MAX),
        1,
    )
    .into_any()))
}
pub fn interrupted(error: &anyhow::Error) -> bool {
    if error
        .downcast_ref::<std::io::Error>()
        .is_some_and(|e| e.kind() == std::io::ErrorKind::Interrupted)
    {
        return true;
    }
    if let Some(dialoguer::Error::IO(e)) = error.downcast_ref::<dialoguer::Error>() {
        if e.kind() == std::io::ErrorKind::Interrupted {
            return true;
        }
    }
    Python::with_gil(|py| {
        error.downcast_ref::<PyErr>().is_some_and(|e| {
            if e.is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>(py) {
                return true;
            }
            let mut context = e.value(py).getattr("__context__").ok();
            for _ in 0..16 {
                let Some(c) = context else { return false };
                if c.is_none() {
                    return false;
                }
                if c.is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>() {
                    return true;
                }
                context = c.getattr("__context__").ok();
            }
            false
        })
    })
}
pub fn import_error(error: &anyhow::Error) -> bool {
    Python::with_gil(|py| {
        error
            .downcast_ref::<PyErr>()
            .is_some_and(|e| e.is_instance_of::<pyo3::exceptions::PyImportError>(py))
    })
}
pub fn control_flow(error: &anyhow::Error) -> bool {
    interrupted(error)
        || Python::with_gil(|py| {
            error
                .downcast_ref::<PyErr>()
                .is_some_and(|e| !e.is_instance_of::<pyo3::exceptions::PyException>(py))
        })
}
pub fn optional<T>(value: Result<T>) -> Result<Option<T>> {
    match value {
        Ok(v) => Ok(Some(v)),
        Err(e) if control_flow(&e) => Err(e),
        Err(_) => Ok(None),
    }
}
pub fn all(py: Python<'_>) -> O<'_> {
    O(PySlice::new(py, 0, isize::MAX, 1).into_any())
}
pub fn norm<'py>(x: &O<'py>, dim: i32) -> Result<O<'py>> {
    let py = x.0.py();
    module(py, "torch.nn.functional")?.m(
        "normalize",
        &[x.clone()],
        &[("p", j(py, 2)?), ("dim", j(py, dim)?)],
    )
}
pub fn empty_cache(py: Python<'_>) -> Result<()> {
    module(py, "gc")?.m("collect", &[], &[])?;
    let torch = module(py, "torch")?;
    let mut cleared = false;
    for backend in ["cuda", "xpu", "mlu", "sdaa", "musa"] {
        if let Ok(device) = torch.a(backend) {
            if device
                .m("is_available", &[], &[])
                .is_ok_and(|v| v.empty().is_ok_and(|e| !e))
            {
                device.m("empty_cache", &[], &[])?;
                cleared = true;
                break;
            }
        }
    }
    if !cleared {
        if let Ok(mps) = torch.a("backends.mps") {
            if !mps.m("is_available", &[], &[])?.empty()? {
                torch.a("mps")?.m("empty_cache", &[], &[])?;
            }
        }
    }
    module(py, "gc")?.m("collect", &[], &[])?;
    Ok(())
}
