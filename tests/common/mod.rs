#![allow(dead_code)]
use anyhow::Result;
use heretic::python::*;
use pyo3::Python;
use serde_json::json;
use std::path::Path;
pub fn fixture(py: Python<'_>, path: &Path) -> Result<()> {
    let tr = module(py, "transformers")?;
    let word=module(py,"tokenizers.models")?.a("WordLevel")?.c(&[],&[("vocab",j(py,json!({"[UNK]":0,"[PAD]":1,"[EOS]":2,"What":3,"is":4,"1":5,"+":6,"?":7,"Hello":8,"world":9,"Tell":10,"me":11,"a":12,"story":13,"How":14,"are":15,"you":16,"today":17,"Good":18,"morning":19,"sun":20,"moon":21,"night":22,"day":23,"sky":24,"blue":25,"green":26,"red":27,"color":28,"the":29,"of":30,"Write":31}))?),("unk_token",j(py,"[UNK]")?)])?;
    let backend = module(py, "tokenizers")?.a("Tokenizer")?.c(&[word], &[])?;
    backend.set(
        "pre_tokenizer",
        &module(py, "tokenizers.pre_tokenizers")?
            .a("Whitespace")?
            .c(&[], &[])?,
    )?;
    let tokenizer = tr.a("PreTrainedTokenizerFast")?.c(
        &[],
        &[
            ("tokenizer_object", backend),
            ("unk_token", j(py, "[UNK]")?),
            ("pad_token", j(py, "[PAD]")?),
            ("eos_token", j(py, "[EOS]")?),
        ],
    )?;
    tokenizer.set("chat_template",&j(py,"{% for m in messages %}{{ m['content'] }}\n{% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}")?)?;
    tokenizer.m("save_pretrained", &[j(py, path.to_string_lossy())?], &[])?;
    module(py, "torch")?.m("manual_seed", &[j(py, 1234)?], &[])?;
    let config = tr.a("LlamaConfig")?.c(
        &[],
        &[
            ("vocab_size", j(py, 32)?),
            ("hidden_size", j(py, 16)?),
            ("intermediate_size", j(py, 32)?),
            ("num_hidden_layers", j(py, 4)?),
            ("num_attention_heads", j(py, 2)?),
            ("num_key_value_heads", j(py, 2)?),
            ("max_position_embeddings", j(py, 256)?),
            ("pad_token_id", j(py, 1)?),
            ("eos_token_id", j(py, 2)?),
            ("bos_token_id", j(py, 0)?),
        ],
    )?;
    let model = tr.a("LlamaForCausalLM")?.c(&[config], &[])?;
    model.m("save_pretrained", &[j(py, path.to_string_lossy())?], &[])?;
    Ok(())
}
pub fn assert_close(py: Python<'_>, a: &O<'_>, b: &O<'_>) -> Result<()> {
    module(py, "torch.testing")?.m(
        "assert_close",
        &[a.clone(), b.clone()],
        &[("rtol", j(py, 0.0)?), ("atol", j(py, 0.0)?)],
    )?;
    Ok(())
}

pub fn variant(py: Python<'_>, path: &Path, config_class: &str, model_class: &str) -> Result<()> {
    fixture(py, path)?;
    let tr = module(py, "transformers")?;
    let mut options = tr
        .a("PretrainedConfig")?
        .m("get_config_dict", &[j(py, path.to_string_lossy())?], &[])?
        .item(0)?
        .json()?;
    options.as_object_mut().unwrap().remove("model_type");
    options["num_experts"] = json!(4);
    options["num_experts_per_tok"] = json!(2);
    options["moe_intermediate_size"] = json!(32);
    let owned = options
        .as_object()
        .unwrap()
        .iter()
        .map(|(key, value)| Ok((key.clone(), j(py, value)?)))
        .collect::<Result<Vec<_>>>()?;
    let kwargs = owned
        .iter()
        .map(|(k, v)| (k.as_str(), v.clone()))
        .collect::<Vec<_>>();
    let config = tr.a(config_class)?.c(&[], &kwargs)?;
    module(py, "torch")?.m("manual_seed", &[j(py, 1234)?], &[])?;
    tr.a(model_class)?.c(&[config], &[])?.m(
        "save_pretrained",
        &[j(py, path.to_string_lossy())?],
        &[],
    )?;
    Ok(())
}
pub fn input(command: &mut std::process::Command, lines: &str) -> Result<std::process::Output> {
    use std::io::Write;
    let mut child = command
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?;
    child.stdin.take().unwrap().write_all(lines.as_bytes())?;
    Ok(child.wait_with_output()?)
}
pub fn load_study<'py>(py: Python<'py>, journal: &Path) -> Result<O<'py>> {
    let backend = module(py, "optuna.storages.journal")?
        .a("JournalFileBackend")?
        .c(&[j(py, journal.to_string_lossy())?], &[])?;
    let storage = module(py, "optuna.storages")?
        .a("JournalStorage")?
        .c(&[backend], &[])?;
    module(py, "optuna")?.m(
        "load_study",
        &[],
        &[("study_name", j(py, "heretic")?), ("storage", storage)],
    )
}
