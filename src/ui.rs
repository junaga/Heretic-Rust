// SPDX-License-Identifier: AGPL-3.0-or-later
//! Native terminal prompts. Numbered input also works in notebook subprocesses and pipes.
use anyhow::{bail, Result};
use std::io::{self, IsTerminal, Write};

fn terminal() -> bool {
    io::stdin().is_terminal() && io::stdout().is_terminal() && !notebook()
}
pub fn notebook() -> bool {
    ["COLAB_GPU", "KAGGLE_KERNEL_RUN_TYPE"]
        .iter()
        .any(|key| std::env::var(key).is_ok_and(|v| !v.is_empty()))
}
fn line(prompt: &str) -> Result<Option<String>> {
    print!("{prompt}");
    io::stdout().flush()?;
    let mut value = String::new();
    if io::stdin().read_line(&mut value)? == 0 {
        return Ok(None);
    }
    Ok(Some(value.trim_end_matches(['\r', '\n']).to_owned()))
}
pub fn select(message: &str, choices: &[String]) -> Result<Option<usize>> {
    if terminal() {
        return Ok(dialoguer::Select::new()
            .with_prompt(message)
            .items(choices)
            .default(0)
            .interact_opt()?);
    }
    println!("{message}");
    for (i, c) in choices.iter().enumerate() {
        println!("[{}] {c}", i + 1);
    }
    loop {
        let Some(value) = line("Enter number (empty to cancel): ")? else {
            return Ok(None);
        };
        if value.is_empty() {
            return Ok(None);
        }
        if let Ok(i) = value.trim().parse::<usize>() {
            if i > 0 && i <= choices.len() {
                return Ok(Some(i - 1));
            }
        }
        println!("Please enter a number between 1 and {}.", choices.len());
    }
}
pub fn text(message: &str, default: &str) -> Result<String> {
    if terminal() {
        return Ok(dialoguer::Input::<String>::new()
            .with_prompt(message)
            .default(default.to_owned())
            .allow_empty(true)
            .interact_text()?);
    }
    let prompt = if default.is_empty() {
        format!("{message}: ")
    } else {
        format!("{message} [{default}]: ")
    };
    let Some(value) = line(&prompt)? else {
        return Ok(String::new());
    };
    Ok(if value.is_empty() {
        default.to_owned()
    } else {
        value
    })
}
pub fn password(message: &str) -> Result<String> {
    if terminal() {
        return Ok(dialoguer::Password::new()
            .with_prompt(message)
            .allow_empty_password(true)
            .interact()?);
    }
    Ok(line(&format!("{message}: "))?.unwrap_or_default())
}
pub fn multiple(message: &str, choices: &[String]) -> Result<Vec<usize>> {
    if terminal() {
        return Ok(dialoguer::MultiSelect::new()
            .with_prompt(message)
            .items(choices)
            .interact()?);
    }
    println!("{message}");
    for (i, c) in choices.iter().enumerate() {
        println!("[{}] {c}", i + 1);
    }
    loop {
        let Some(value) = line("Comma-separated numbers (empty to cancel): ")? else {
            return Ok(Vec::new());
        };
        if value.is_empty() {
            return Ok(Vec::new());
        }
        match parse_multiple(&value, choices.len()) {
            Ok(v) => return Ok(v),
            Err(e) => println!("{e}"),
        }
    }
}
pub fn parse_multiple(value: &str, count: usize) -> Result<Vec<usize>> {
    let mut selected = Vec::new();
    for s in value.split(',') {
        let i = s.trim().parse::<usize>()?;
        if i == 0 || i > count {
            bail!("Please choose numbers between 1 and {count}")
        };
        if !selected.contains(&(i - 1)) {
            selected.push(i - 1);
        }
    }
    Ok(selected)
}
#[cfg(test)]
mod tests {
    #[test]
    fn multiselect() {
        assert_eq!(super::parse_multiple("1,3,1", 3).unwrap(), vec![0, 2]);
        assert!(super::parse_multiple("0", 3).is_err());
    }
}
