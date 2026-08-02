use std::path::PathBuf;

use color_eyre::eyre::{bail, Result, WrapErr};
use polars::prelude::*;

const MU: f64 = 1.29e-8;

// struct to hold number of differences and "length"
// for a single pairwise intersection of segments
#[derive(Debug, Copy, Clone)]
pub struct SegmentDivergence {
    pub k: f64,
    pub mu: f64,    // mutation rate per segment, i.e. "length"
    pub count: u32, // count of data rows with these values of k and mu
}

pub fn read_divergences(path: PathBuf) -> Result<Box<[SegmentDivergence]>> {
    let input = PlRefPath::try_from_path(&path)
        .wrap_err_with(|| format!("invalid input path {}", path.display()))?;

    let schema = LazyFrame::scan_parquet(input.clone(), Default::default())?.collect_schema()?;

    // to accomodate more input files, we will accept both "diff" (new) and "n_diff" (old) columns names
    let diff_column = if schema.contains("n_diff") {
        "n_diff"
    } else if schema.contains("diff") {
        "diff"
    } else {
        bail!("no divergence column found in input table! please ensure you have a column named 'n_diff' or 'diff'")
    };

    let mut divs = LazyFrame::scan_parquet(input, Default::default())?
        .sort(["sa", "sb", "chrom"], Default::default())
        .with_columns([
            col(diff_column).cast(DataType::Float64),
            col("intersection_len").cast(DataType::Float64),
        ])
        .collect()?;

    let sa = divs.column("sa")?.str()?;
    let sb = divs.column("sb")?.str()?;

    let pair: Vec<_> = sa
        .iter()
        .zip(sb.iter())
        .map(|(x, y)| format!("{};{}", x.unwrap(), y.unwrap()))
        .collect();
    let pair = Column::new("pair_label".into(), pair);

    let divs_seg = divs.with_column(pair)?.clone().lazy();

    let divs_counts = divs_seg
        .group_by([diff_column, "intersection_len"])
        .agg([len().alias("count")])
        .collect()?;

    let diffs = divs_counts.column(diff_column)?.f64()?;
    let lengths = divs_counts.column("intersection_len")?.f64()?;
    let counts = divs_counts.column("count")?.u32()?;

    let ans = diffs
        .iter()
        .zip(lengths.iter())
        .zip(counts.iter())
        .map(|((k, l), count)| {
            let (k, l, count) = (k.unwrap(), l.unwrap(), count.unwrap());
            // convert intersection length from basepairs to mutation units
            let mu = l * MU;

            SegmentDivergence { k, mu, count }
        })
        .collect();

    Ok(ans)
}

pub fn bootstrap_divergences(path: PathBuf, seed: u64) -> Result<Box<[SegmentDivergence]>> {
    let input = PlRefPath::try_from_path(&path)
        .wrap_err_with(|| format!("invalid input path: {}", path.display()))?;

    let schema = LazyFrame::scan_parquet(input.clone(), Default::default())?.collect_schema()?;

    // to accomodate more input files, we will accept both "diff" (new) and "n_diff" (old) columns names
    let diff_column = if schema.contains("n_diff") {
        "n_diff"
    } else if schema.contains("diff") {
        "diff"
    } else {
        bail!("no divergence column found in input table! please ensure you have a column named 'n_diff' or 'diff'")
    };

    let mut divs = LazyFrame::scan_parquet(input, Default::default())?
        .sort(["sa", "sb", "chrom"], Default::default())
        .with_columns([
            col(diff_column).cast(DataType::Float64),
            col("intersection_len").cast(DataType::Float64),
        ])
        .collect()?;

    let sa = divs.column("sa")?.str()?;
    let sb = divs.column("sb")?.str()?;

    let pair: Vec<_> = sa
        .iter()
        .zip(sb.iter())
        .map(|(x, y)| format!("{};{}", x.unwrap(), y.unwrap()))
        .collect();
    let pair = Column::new("pair_label".into(), pair);

    let divs_seg = divs.with_column(pair)?.clone().lazy();

    let divs_seg = divs_seg.collect()?.sample_frac(
        &Series::new("frac".into(), &[1.0f64]),
        true,
        false,
        Some(seed),
    )?;

    let divs_counts = divs_seg
        .lazy()
        .group_by([diff_column, "intersection_len"])
        .agg([len().alias("count")])
        .collect()?;

    let diffs = divs_counts.column(diff_column)?.f64()?;
    let lengths = divs_counts.column("intersection_len")?.f64()?;
    let counts = divs_counts.column("count")?.u32()?;

    let ans = diffs
        .iter()
        .zip(lengths.iter())
        .zip(counts.iter())
        .map(|((k, l), count)| {
            let (k, l, count) = (k.unwrap(), l.unwrap(), count.unwrap());
            // convert intersection length from basepairs to mutation units
            let mu = l * MU;

            SegmentDivergence { k, mu, count }
        })
        .collect();

    Ok(ans)
}
