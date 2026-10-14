// SPDX-License-Identifier: AGPL-3.0-or-later
// Optional residual analysis ported from Heretic v1.4.0.
use crate::{model::Model, python::*};
use anyhow::Result;
use serde_json::json;
use std::path::Path;

pub fn analyze(m: &Model<'_>, good: &O<'_>, bad: &O<'_>) -> Result<()> {
    if m.settings.flag("print_residual_geometry") {
        if let Err(e) = geometry(m, good, bad) {
            eprintln!("Residual geometry unavailable: {e:#}. Install geom-median and scikit-learn for research support.");
        }
    }
    if m.settings.flag("plot_residuals") {
        if let Err(e) = plots(m, good, bad) {
            eprintln!("Residual plots unavailable: {e:#}. Install geom-median, pacmap, imageio and matplotlib for research support.");
        }
    }
    Ok(())
}
fn geometry(m: &Model<'_>, good: &O<'_>, bad: &O<'_>) -> Result<()> {
    let py = m.py;
    let torch = module(py, "torch")?;
    let median = module(py, "geom_median.torch")?.a("compute_geometric_median")?;
    let metrics = module(py, "sklearn.metrics")?;
    let g = good.m("mean", &[], &[("dim", j(py, 0)?)])?;
    let b = bad.m("mean", &[], &[("dim", j(py, 0)?)])?;
    let n = m.layers()?.len()?;
    let mut gm = Vec::new();
    let mut bm = Vec::new();
    for i in 0..=n {
        for (data, target) in [(good, &mut gm), (bad, &mut bm)] {
            let residuals = data
                .tensor_slice(&[all(py), j(py, i)?, all(py)])?
                .m("detach", &[], &[])?
                .m("cpu", &[], &[])?;
            target.push(median.c(&[residuals], &[])?.a("median")?);
        }
    }
    let gs = torch.m("stack", &[list(py, &gm)?], &[])?;
    let bs = torch.m("stack", &[list(py, &bm)?], &[])?;
    let r = b.sub(&g)?;
    let rs = bs.sub(&gs)?;
    let mut columns = Vec::new();
    for (a, b) in [
        (&g, &b),
        (&gs, &bs),
        (&g, &r),
        (&gs, &rs),
        (&b, &r),
        (&bs, &rs),
    ] {
        columns.push(module(py, "torch.nn.functional")?.m(
            "cosine_similarity",
            &[a.clone(), b.clone()],
            &[("dim", j(py, -1)?)],
        )?);
    }
    for t in [&g, &gs, &b, &bs, &r, &rs] {
        columns.push(module(py, "torch.linalg")?.m(
            "vector_norm",
            &[t.clone()],
            &[("dim", j(py, -1)?)],
        )?);
    }
    let residuals = torch
        .m(
            "cat",
            &[list(py, &[good.clone(), bad.clone()])?],
            &[("dim", j(py, 0)?)],
        )?
        .m("detach", &[], &[])?
        .m("cpu", &[], &[])?
        .m("numpy", &[], &[])?;
    let labels: Vec<_> = std::iter::repeat_n(0, good.len()?)
        .chain(std::iter::repeat_n(1, bad.len()?))
        .collect();
    println!("Layer  S(g,b)  S(g*,b*)  S(g,r)  S(g*,r*)  S(b,r)  S(b*,r*)  |g|  |g*|  |b|  |b*|  |r|  |r*|  Silh");
    for i in 1..=n {
        let mut row = format!("{i:>5}");
        for c in &columns {
            row += &format!("  {:.4}", c.item(i as isize)?.m("item", &[], &[])?.f64()?);
        }
        let points = residuals.tensor_slice(&[all(py), j(py, i)?, all(py)])?;
        let silhouette = metrics
            .m("silhouette_score", &[points, j(py, &labels)?], &[])?
            .f64()?;
        println!("{row}  {silhouette:.4}");
    }
    println!("g/b: good/bad means; g*/b*: geometric medians; r=b-g; r*=b*-g*; S: cosine similarity; |x|: L2 norm; Silh: silhouette coefficient.");
    Ok(())
}
fn plot(m: &Model<'_>, plt: &O<'_>, file: &Path, layer: usize, g: &O<'_>, b: &O<'_>) -> Result<()> {
    let py = m.py;
    let pair = plt.m("subplots", &[], &[("figsize", j(py, [8, 6])?)])?;
    let fig = pair.item(0)?;
    let ax = pair.item(1)?;
    for (points, spec) in [
        (g, m.settings.get("good_prompts")),
        (b, m.settings.get("bad_prompts")),
    ] {
        ax.m(
            "scatter",
            &[
                points.tensor_slice(&[all(py), j(py, 0)?])?,
                points.tensor_slice(&[all(py), j(py, 1)?])?,
            ],
            &[
                ("s", j(py, 10)?),
                ("c", j(py, &spec["residual_plot_color"])?),
                ("alpha", j(py, 0.5)?),
                ("label", j(py, &spec["residual_plot_label"])?),
            ],
        )?;
    }
    ax.m(
        "set_title",
        &[j(py, m.settings.get("residual_plot_title"))?],
        &[("pad", j(py, 11)?)],
    )?;
    ax.m("legend", &[], &[("loc", j(py, "upper right")?)])?;
    ax.m("grid", &[j(py, false)?], &[])?;
    for method in ["set_xticks", "set_yticks"] {
        ax.m(method, &[j(py, Vec::<usize>::new())?], &[])?;
    }
    fig.m(
        "text",
        &[j(py, 0.018)?, j(py, 0.02)?, j(py, m.settings.get("model"))?],
        &[
            ("ha", j(py, "left")?),
            ("va", j(py, "bottom")?),
            ("fontsize", j(py, 12)?),
        ],
    )?;
    fig.m(
        "text",
        &[
            j(py, 0.982)?,
            j(py, 0.02)?,
            j(py, format!("Layer {layer:03}"))?,
        ],
        &[
            ("ha", j(py, "right")?),
            ("va", j(py, "bottom")?),
            ("fontsize", j(py, 12)?),
        ],
    )?;
    fig.m("tight_layout", &[], &[])?;
    fig.m("subplots_adjust", &[], &[("bottom", j(py, 0.08)?)])?;
    fig.m(
        "savefig",
        &[j(py, file.to_string_lossy())?],
        &[("dpi", j(py, 100)?)],
    )?;
    plt.m("close", &[fig], &[])?;
    Ok(())
}
fn plots(m: &Model<'_>, good: &O<'_>, bad: &O<'_>) -> Result<()> {
    let py = m.py;
    let np = module(py, "numpy")?;
    let plt = module(py, "matplotlib.pyplot")?;
    let iio = module(py, "imageio.v3")?;
    let pacmap = module(py, "pacmap")?.a("PaCMAP")?;
    let median = module(py, "geom_median.numpy")?.a("compute_geometric_median")?;
    let mut init = j(py, serde_json::Value::Null)?;
    let mut projections = Vec::new();
    let count = m.layers()?.len()?;
    for i in 1..=count {
        let g = good
            .tensor_slice(&[all(py), j(py, i)?, all(py)])?
            .m("detach", &[], &[])?
            .m("cpu", &[], &[])?
            .m("numpy", &[], &[])?;
        let b = bad
            .tensor_slice(&[all(py), j(py, i)?, all(py)])?
            .m("detach", &[], &[])?
            .m("cpu", &[], &[])?
            .m("numpy", &[], &[])?;
        let data = np.m("vstack", &[tuple(py, &[g.clone(), b])?], &[])?;
        let embedding = pacmap.c(
            &[],
            &[("n_components", j(py, 2)?), ("n_neighbors", j(py, 30)?)],
        )?;
        let points = embedding.m("fit_transform", &[data], &[("init", init)])?;
        init = points.clone();
        let n = g.len()? as isize;
        let gp = points.idx(&slice(py, None, Some(n))?)?;
        let bp = points.idx(&slice(py, Some(n), None)?)?;
        let direction = median
            .c(&[bp], &[])?
            .a("median")?
            .sub(&median.c(&[gp], &[])?.a("median")?)?;
        let angle = -np
            .m("arctan2", &[direction.item(1)?, direction.item(0)?], &[])?
            .f64()?;
        let (sin, cos) = angle.sin_cos();
        let rotation = np.m("array", &[j(py, json!([[cos, -sin], [sin, cos]]))?], &[])?;
        let rotated = points.mat(&rotation.a("T")?)?;
        projections.push((
            rotated.idx(&slice(py, None, Some(n))?)?,
            rotated.idx(&slice(py, Some(n), None)?)?,
        ));
    }
    plt.a("style")?
        .m("use", &[j(py, m.settings.get("residual_plot_style"))?], &[])?;
    let base = Path::new(&m.settings.string("residual_plot_path"))
        .join(m.settings.string("model").replace(['/', '\\'], "_"));
    std::fs::create_dir_all(&base)?;
    let mut images = Vec::new();
    let mut durations = Vec::new();
    for (index, (g, b)) in projections.iter().enumerate() {
        let file = base.join(format!("layer_{:03}.png", index + 1));
        plot(m, &plt, &file, index + 1, g, b)?;
        images.push(iio.m("imread", &[j(py, file.to_string_lossy())?], &[])?);
        durations.push(1000);
        if let Some((nextg, nextb)) = projections.get(index + 1) {
            for frame in 1..20 {
                let t = j(py, frame as f64 / 20.0)?;
                let gi = g.add(&nextg.sub(g)?.mul(&t)?)?;
                let bi = b.add(&nextb.sub(b)?.mul(&t)?)?;
                let file = base.join(format!("layer_{:03}_frame_{frame:03}.png", index + 1));
                plot(m, &plt, &file, index + 1, &gi, &bi)?;
                images.push(iio.m("imread", &[j(py, file.to_string_lossy())?], &[])?);
                durations.push(50);
                std::fs::remove_file(file)?;
            }
        }
    }
    iio.m(
        "imwrite",
        &[
            j(py, base.join("animation.gif").to_string_lossy())?,
            list(py, &images)?,
        ],
        &[("duration", j(py, durations)?), ("loop", j(py, 0)?)],
    )?;
    println!("Plots saved to {}", base.display());
    Ok(())
}
