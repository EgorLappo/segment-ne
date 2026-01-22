use clap::{Parser, Subcommand};
use color_eyre::eyre::{bail, Result};
use itertools::Itertools;
use std::io::Write;
use std::path::PathBuf;

mod data;
mod lik;
mod mcmc;
mod optim;
mod parameter;

fn main() -> Result<()> {
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

            let mut samples = Vec::new();

            for h in handles.into_iter() {
                let chain_samples = h.join().expect("could not join on a thread");
                samples.extend_from_slice(&chain_samples);
            }

            let mut file = std::fs::File::create(opts.output)?;

            for (ll, step_vals) in samples.iter() {
                writeln!(file, "{:?},{}", ll, step_vals.iter().format(" "))?;
            }
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
