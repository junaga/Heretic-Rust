// SPDX-License-Identifier: AGPL-3.0-or-later
// Algorithms ported from Heretic v1.4.0, Copyright Philipp Emanuel Weidmann + contributors.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Prompt {
    pub system: String,
    pub user: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AbliterationParameters {
    pub max_weight: f64,
    pub max_weight_position: f64,
    pub min_weight: f64,
    pub min_weight_distance: f64,
}
impl AbliterationParameters {
    pub fn weight(&self, layer: usize) -> Option<f64> {
        let distance = (layer as f64 - self.max_weight_position).abs();
        if distance > self.min_weight_distance {
            None
        } else {
            Some(
                self.max_weight
                    + distance / self.min_weight_distance * (self.min_weight - self.max_weight),
            )
        }
    }
}
pub type Parameters = BTreeMap<String, AbliterationParameters>;
fn python_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}
pub fn is_refusal(response: &str, markers: &[String]) -> bool {
    if response.chars().all(python_space) {
        return true;
    }
    let s = response
        .to_lowercase()
        .replace('*', "")
        .replace('’', "'")
        .split(python_space)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    markers.iter().any(|m| s.contains(&m.to_lowercase()))
}
pub fn score(kl: f64, refusals: usize, base: usize, scale: f64, target: f64) -> (f64, f64) {
    let refusal = if base > 0 {
        refusals as f64 / base as f64
    } else {
        refusals as f64
    };
    (
        if kl >= target {
            kl / scale
        } else {
            refusal * target / scale
        },
        refusal,
    )
}
pub fn common_prefix(items: &[String]) -> String {
    let Some(first) = items.iter().min() else {
        return String::new();
    };
    // The common prefix of a set is the common prefix of its lexical extremes.
    // Scan Unicode scalars once instead of repeatedly indexing UTF-8 strings.
    let last = items.iter().max().unwrap();
    first
        .chars()
        .zip(last.chars())
        .take_while(|(a, b)| a == b)
        .map(|(c, _)| c)
        .collect::<String>()
        .trim_end_matches(' ')
        .to_owned()
}
pub fn checkpoint_name(model: &str) -> String {
    model
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c == '-' {
                c.to_string()
            } else {
                "--".to_owned()
            }
        })
        .collect::<String>()
        + ".jsonl"
}
pub fn pareto_indices(metrics: &[(usize, f64)]) -> Vec<usize> {
    let mut order: (Vec<usize>, f64) = ((0..metrics.len()).collect(), f64::INFINITY);
    order.0.sort_by(|&a, &b| {
        metrics[a]
            .0
            .cmp(&metrics[b].0)
            .then(metrics[a].1.total_cmp(&metrics[b].1))
    });
    order
        .0
        .into_iter()
        .filter(|&i| {
            if metrics[i].1 < order.1 {
                order.1 = metrics[i].1;
                true
            } else {
                false
            }
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn refusal_normalization() {
        let m = vec!["i will not".into(), "won't".into()];
        assert!(is_refusal("I *will\nnot* comply", &m));
        assert!(is_refusal("I won’t", &m));
        assert!(is_refusal("I\u{1c}will not", &m));
        assert!(is_refusal("  ", &m));
        assert!(!is_refusal("Here you go", &m));
    }
    #[test]
    fn objectives() {
        assert_eq!(score(0.0, 20, 100, 1.0, 0.01), (0.002, 0.2));
        assert_eq!(score(0.2, 2, 0, 2.0, 0.01), (0.1, 2.0));
    }
    #[test]
    fn kernel() {
        let p = AbliterationParameters {
            max_weight: 1.0,
            max_weight_position: 2.0,
            min_weight: 0.0,
            min_weight_distance: 2.0,
        };
        assert_eq!(p.weight(2), Some(1.0));
        assert_eq!(p.weight(0), Some(0.0));
        assert_eq!(p.weight(5), None);
    }
    #[test]
    fn pareto() {
        assert_eq!(
            pareto_indices(&[(2, 0.2), (1, 0.3), (2, 0.4), (3, 0.1)]),
            vec![1, 0, 3]
        );
    }
    #[test]
    fn unicode_prefix() {
        assert_eq!(
            common_prefix(&["é <think> x".into(), "é <think> y".into()]),
            "é <think>"
        );
    }
}
