use std::path::PathBuf;

use color_eyre::eyre::Result;
use polars::prelude::*;

const MU: f64 = 1.29e-8;

// struct to hold number of differences and "length"
// for a single pairwise intersection of segments
#[derive(Debug, Copy, Clone)]
pub struct SegmentDivergence {
    pub k: f64,
    pub mu: f64, // mutation rate per segment, i.e. "length"
}

pub fn read_divergences(path: PathBuf, fast: bool) -> Result<Box<[SegmentDivergence]>> {
    let input = Arc::from(path);
    let input = PlPath::Local(input);

    let mut divs = LazyFrame::scan_parquet(input, Default::default())?
        .sort(["sa", "sb", "chrom"], Default::default())
        .with_columns([
            col("n_diff").cast(DataType::Float64),
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
    let pair = Series::new("pair_label".into(), pair);

    let divs_seg = divs.with_column(pair)?.clone().lazy();

    let divs_seg = if fast {
        // if we want to make it faster, sum the observations by chromosome
        divs_seg
            .group_by(["pair_label", "chrom"])
            .agg([col("intersection_len").sum(), col("n_diff").sum()])
    } else {
        divs_seg
    };

    let divs_seg = divs_seg.filter(col("n_diff").gt(lit(0.0))).collect()?;

    let ans = divs_seg
        .column("n_diff")?
        .f64()?
        .iter()
        .map(|x| x.unwrap())
        .zip(
            divs_seg
                .column("intersection_len")?
                .f64()?
                .iter()
                .map(|x| x.unwrap() * MU),
        )
        .map(|(k, mu)| SegmentDivergence { k, mu })
        .collect();

    Ok(ans)
}

pub fn bootstrap_divergences(
    path: PathBuf,
    fast: bool,
    seed: u64,
) -> Result<Box<[SegmentDivergence]>> {
    let input = Arc::from(path);
    let input = PlPath::Local(input);

    let mut divs = LazyFrame::scan_parquet(input, Default::default())?
        .sort(["sa", "sb", "chrom"], Default::default())
        .with_columns([
            col("n_diff").cast(DataType::Float64),
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
    let pair = Series::new("pair_label".into(), pair);

    let divs_seg = divs.with_column(pair)?.clone().lazy();

    let divs_seg = if fast {
        // if we want to make it faster, sum the observations by chromosome
        divs_seg
            .group_by(["pair_label", "chrom"])
            .agg([col("intersection_len").sum(), col("n_diff").sum()])
    } else {
        divs_seg
    };
    let divs_seg = divs_seg
        .filter(col("n_diff").gt(lit(0.0)))
        .collect()?
        .sample_frac(
            &Series::new("frac".into(), &[1.0f64]),
            true,
            false,
            Some(seed),
        )?;

    let ans = divs_seg
        .column("n_diff")?
        .f64()?
        .iter()
        .map(|x| x.unwrap())
        .zip(
            divs_seg
                .column("intersection_len")?
                .f64()?
                .iter()
                .map(|x| x.unwrap() * MU),
        )
        .map(|(k, mu)| SegmentDivergence { k, mu })
        .collect();

    Ok(ans)
}
