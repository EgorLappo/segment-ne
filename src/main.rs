use clap::{Parser, Subcommand};
use color_eyre::eyre::{bail, Result};
use itertools::Itertools;
use polars::prelude::*;
use std::io::Write;
use std::path::PathBuf;

mod data;
mod mcmc;
mod observation;
mod optim;
mod parameter;
mod skyline;

fn main() -> Result<()> {
    // init error handling
    color_eyre::install()?;

    // build logger right away
    let logger =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).build();
    // the logger ^ is not initialized as we also need to make it
    // work with the progress bar. so, create the bar and init logger
    let bar = indicatif::MultiProgress::new();
    indicatif_log_bridge::LogWrapper::new(bar.clone(), logger).try_init()?;

    // finally, we can parse the args
    let opts = Opts::parse();

    // parameter validation
    let parameters = crate::parameter::Parameters::new(
        &opts.pop_sizes,
        &opts.change_times,
        opts.admixture_fraction,
        opts.admixture_index,
    )?;

    let data = if let Some(boot) = opts.boot {
        data::bootstrap_divergences(opts.input, opts.fast, boot)?
    } else {
        data::read_divergences(opts.input, opts.fast)?
    };

    match opts.command {
        Command::Optim => {
            let total_fit_params = parameters.t.num_fit() + parameters.c.num_fit();
            // optimization only works with variable population size (s)
            if total_fit_params == 0 {
                bail!("no parameters to be fit were provided. please annotate them with '~'")
            }

            // see if the problem is single- or multi-variable
            if total_fit_params == 1 {
                let result = optim::optimize(&data, parameters)?;
                println!("{:?}", result);

                let mut file = std::fs::File::create(opts.output)?;
                writeln!(file, "{:?}", result)?;
            } else {
                let (n, t) = optim::optimize_multivariable(&data, parameters)?;
                println!("{:?}", n);
                println!("{:?}", t);

                let mut file = std::fs::File::create(opts.output)?;
                writeln!(file, "{:?}", n.iter().format(" "))?;
                writeln!(file, "{:?}", t.iter().format(" "))?;
            }
        }
        Command::Sample {
            seed,
            chains,
            warmup,
            sampling,
        } => {
            use mcmc::*;

            let handles: Vec<_> = (0..chains)
                .map(|_| {
                    let data = data.clone();
                    let parameters = parameters.clone();
                    let bar = bar.clone();
                    std::thread::spawn(move || {
                        let mut chain = Chain::new(&data, parameters);

                        chain.run(warmup, sampling, seed, bar)
                    })
                })
                .collect();

            let mut chain_dfs = Vec::new();

            for (i, h) in handles.into_iter().enumerate() {
                let chain_samples = h.join().expect("could not join on a thread");

                let ll = Column::new("loglik".into(), chain_samples.0);

                let n_samples: Vec<_> = chain_samples
                    .1
                    .into_iter()
                    .map(|x| Series::new("nsamp".into(), x))
                    .collect();
                let n_samples = Column::new("n".into(), n_samples);

                let t_samples: Vec<_> = chain_samples
                    .2
                    .into_iter()
                    .map(|x| Series::new("tsamp".into(), x))
                    .collect();
                let t_samples = Column::new("t".into(), t_samples);

                let chain_df = DataFrame::new(vec![ll, n_samples, t_samples])?;

                let chain_df = chain_df.lazy().with_column(lit(i as u64).alias("chain"));
                chain_dfs.push(chain_df);
            }

            let mut df = concat(chain_dfs, UnionArgs::default())?.collect()?;

            let mut out_path = opts.output;
            out_path.set_extension("parquet");
            let out_file = std::fs::File::create(out_path)?;
            ParquetWriter::new(out_file).finish(&mut df)?;
        }
        Command::Skyline {
            num_intervals,
            t_scale,
            alpha,
            seed,
            chains,
            warmup,
            sampling,
        } => {
            use skyline::*;

            if t_scale <= 1. {
                bail!("value of t_scale={:?} invalid. must be > 1", t_scale);
            }

            if alpha <= 0. {
                bail!("value of alpha={:?} invalid. must be positive", alpha);
            }

            let parameters = parameters.expand_skyline(num_intervals)?;

            let handles: Vec<_> = (0..chains)
                .map(|_| {
                    let data = data.clone();
                    let parameters = parameters.clone();
                    let bar = bar.clone();
                    std::thread::spawn(move || {
                        let mut chain =
                            SkylineChain::new(&data, parameters, num_intervals, t_scale, alpha);

                        chain.run(warmup, sampling, seed, bar)
                    })
                })
                .collect();

            let mut chain_dfs = Vec::new();

            for (i, h) in handles.into_iter().enumerate() {
                let chain_samples = h.join().expect("could not join on a thread");

                let ll = Column::new("loglik".into(), chain_samples.0);

                let n_samples: Vec<_> = chain_samples
                    .1
                    .into_iter()
                    .map(|x| Series::new("nsamp".into(), x))
                    .collect();
                let n_samples = Column::new("n".into(), n_samples);

                let t_samples: Vec<_> = chain_samples
                    .2
                    .into_iter()
                    .map(|x| Series::new("tsamp".into(), x))
                    .collect();
                let t_samples = Column::new("t".into(), t_samples);

                let chain_df = DataFrame::new(vec![ll, n_samples, t_samples])?;

                let chain_df = chain_df.lazy().with_column(lit(i as u64).alias("chain"));
                chain_dfs.push(chain_df);
            }

            let mut df = concat(chain_dfs, UnionArgs::default())?.collect()?;

            let mut out_path = opts.output;
            out_path.set_extension("parquet");
            let out_file = std::fs::File::create(out_path)?;
            ParquetWriter::new(out_file).finish(&mut df)?;
        }
    };

    Ok(())
}

#[derive(Debug, Clone, Parser)]
#[command(version, about = "Sample from Ewens distibution conditional on number of observed alleles.", long_about = None)]
struct Opts {
    #[arg(short, long, value_name = "IN", help = "input parquet file(s)")]
    input: PathBuf,
    #[arg(
        short,
        long,
        default_value = "./segment-ne-out",
        value_name = "OUT",
        help = "output file"
    )]
    output: PathBuf,
    #[arg(
        short = 'a',
        long = "admixture_fraction",
        value_name = "FRAC",
        default_value_t = 1.0,
        help = "Admixture fraction from the target population"
    )]
    admixture_fraction: f64,
    #[arg(
        short = 'l',
        long = "admixture_segment",
        value_name = "INDEX",
        default_value_t = 1,
        help = "index of breakpoint at which admixture happens (1-based; l=m means admixture after the mth constant segment"
    )]
    admixture_index: usize,
    #[arg(
        short = 'n',
        long = "sizes",
        value_name = "LIST",
        help = "population sizes"
    )]
    pop_sizes: String,
    #[arg(
        short = 't',
        long = "times",
        value_name = "LIST",
        help = "population size change times"
    )]
    change_times: String,
    #[arg(
        short,
        long,
        value_name = "SEED",
        help = "pass a seed to bootstrap the rows of the input table"
    )]
    boot: Option<u64>,
    #[arg(
        short,
        long,
        default_value_t = false,
        help = "use fast computation (aggregate segments on each chromosome)"
    )]
    fast: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Clone, Subcommand)]
enum Command {
    Optim,
    Sample {
        #[arg(
            short,
            long,
            value_name = "SEED",
            help = "random_seed",
            default_value_t = 231
        )]
        seed: u64,
        #[arg(
            short = 'p',
            long,
            value_name = "CHAINS",
            help = "number of parallel chains",
            default_value_t = 1
        )]
        chains: usize,
        #[arg(
            value_name = "STEPS",
            help = "number of warmup steps",
            default_value_t = 1000
        )]
        warmup: usize,
        #[arg(
            value_name = "STEPS",
            help = "number of sampling steps",
            default_value_t = 5000
        )]
        sampling: usize,
    },
    Skyline {
        #[arg(short = 'r', long, help = "number of intervals", default_value_t = 4)]
        num_intervals: usize,
        #[arg(
            short = 'c',
            long,
            help = "expected ratio of breakpoint times",
            default_value_t = 2.
        )]
        t_scale: f64,
        #[arg(
            short = 'A',
            help = "variance of interval lenghts",
            default_value_t = 7.
        )]
        alpha: f64,
        #[arg(
            short,
            long,
            value_name = "SEED",
            help = "random_seed",
            default_value_t = 231
        )]
        seed: u64,
        #[arg(
            short = 'p',
            long,
            value_name = "CHAINS",
            help = "number of parallel chains",
            default_value_t = 1
        )]
        chains: usize,
        #[arg(
            value_name = "STEPS",
            help = "number of warmup steps",
            default_value_t = 1000
        )]
        warmup: usize,
        #[arg(
            value_name = "STEPS",
            help = "number of sampling steps",
            default_value_t = 5000
        )]
        sampling: usize,
    },
}
