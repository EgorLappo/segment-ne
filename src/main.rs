use clap::{Parser, Subcommand};
use color_eyre::eyre::Result;
use itertools::Itertools;
use polars::prelude::*;
use std::io::Write;
use std::path::PathBuf;

use crate::mcmc::Observation;

mod data;
mod mcmc;
mod optim;
mod parameter;

fn main() -> Result<()> {
    // debug code
    let ns = vec![10000., 1250., 3400., 18500., 10000.];
    let ts = vec![0., 1800., 5000., 18965., 100000.];
    let k = 3.;
    let mu = 1e-4;

    let obs = Observation::new(k, mu);

    println!("new: {}", obs.lpdf(ns.iter(), ts.iter()));
    println!("old: {}", optim::lik::k_lpdf(k, &ns, &ts, mu));

    let logger =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).build();
    color_eyre::install()?;
    let opts = Opts::parse();

    // parameter validation
    let parameters = crate::parameter::Parameters::new(&opts.pop_sizes, &opts.change_times)?;

    match opts.command {
        Command::Optim => {
            // optimization only works with variable population size (s)
            if !parameters.t.num_fit() > 0 {
                log::warn!(
                    "optimizing time variables is not yet supported, will use them as fixed"
                );
            }

            let data = data::read_divergences(opts.input)?;

            // see if the problem is single- or multi-variable
            if parameters.n.num_fit() == 1 {
                let result = optim::optimize(&data, parameters)?;
                println!("{:?}", result);

                let mut file = std::fs::File::create(opts.output)?;
                writeln!(file, "{:?}", result)?;
            } else {
                let result = optim::optimize_multivariable(&data, parameters)?;
                println!("{:?}", result);

                let mut file = std::fs::File::create(opts.output)?;
                writeln!(file, "{:?}", result.iter().format(" "))?;
            }
        }
        Command::Boot { seed } => {
            // optimization only works with variable population size (s)
            if !parameters.t.num_fit() > 0 {
                log::warn!(
                    "optimizing time variables is not yet supported, will use them as fixed"
                );
            }

            let data = data::bootstrap_divergences(opts.input, seed)?;

            // see if the problem is single- or multi-variable
            if parameters.n.num_fit() == 1 {
                let result = optim::optimize(&data, parameters)?;
                println!("{:?}", result);

                let mut file = std::fs::File::create(opts.output)?;
                writeln!(file, "{:?}", result)?;
            } else {
                let result = optim::optimize_multivariable(&data, parameters)?;
                println!("{:?}", result);

                let mut file = std::fs::File::create(opts.output)?;
                writeln!(file, "{:?}", result.iter().format(" "))?;
            }
        }
        Command::Sample {
            seed,
            chains,
            warmup,
            sampling,
        } => {
            use mcmc::*;

            let bar = indicatif::MultiProgress::new();
            indicatif_log_bridge::LogWrapper::new(bar.clone(), logger).try_init()?;

            let data = data::read_divergences(opts.input)?;

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
            let out_file = std::fs::File::create_new(out_path)?;
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
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Clone, Subcommand)]
enum Command {
    Optim,
    Boot {
        #[arg(
            short,
            long,
            value_name = "SEED",
            help = "random_seed",
            default_value_t = 231
        )]
        seed: u64,
    },
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
}
