// SPDX-License-Identifier: AGPL-3.0-or-later
//! Thin CPython FFI. Heretic algorithms live in Rust; imported modules are third-party dependencies.
use anyhow::Result;
use pyo3::{
    prelude::*,
    types::{PyDict, PySlice, PyTuple},
};
use serde::Serialize;
use serde_json::Value;

#[derive(Clone)]
pub struct O<'py>(pub Bound<'py, PyAny>);
impl<'py> O<'py> {
    pub fn a(&self, path: &str) -> Result<Self> {
        let mut x = self.clone();
        for name in path.split('.') {
            x = Self(x.0.getattr(name)?);
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
        Ok(Self(self.0.call(tuple, Some(&d))?))
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
        let s = module(self.0.py(), "json")?
            .m("dumps", &[self.clone()], &[])?
            .str()?;
        Ok(serde_json::from_str(&s)?)
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
    pub fn tensor_slice(&self, parts: &[Self]) -> Result<Self> {
        self.idx(&tuple(self.0.py(), parts)?)
    }
}
pub fn module<'py>(py: Python<'py>, name: &str) -> Result<O<'py>> {
    Ok(O(py.import(name)?.into_any()))
}
pub fn j<'py>(py: Python<'py>, v: impl Serialize) -> Result<O<'py>> {
    let s = serde_json::to_string(&v)?;
    Ok(O(py.import("json")?.call_method1("loads", (s,))?))
}
pub fn tuple<'py>(py: Python<'py>, v: &[O<'py>]) -> Result<O<'py>> {
    Ok(O(PyTuple::new(py, v.iter().map(|x| &x.0))?.into_any()))
}
pub fn list<'py>(py: Python<'py>, v: &[O<'py>]) -> Result<O<'py>> {
    Ok(module(py, "builtins")?
        .a("list")?
        .c(&[tuple(py, v)?], &[])?)
}
pub fn slice<'py>(py: Python<'py>, start: Option<isize>, stop: Option<isize>) -> Result<O<'py>> {
    module(py, "builtins")?
        .a("slice")?
        .c(&[j(py, start)?, j(py, stop)?], &[])
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
