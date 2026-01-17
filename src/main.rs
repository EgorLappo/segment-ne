use clap::{Parser, Subcommand};
use color_eyre::eyre::{bail, Result};
use std::io::Write;
use std::path::PathBuf;

mod data;
mod lik;
mod mcmc;
mod optim;
mod parameter;

fn main() -> Result<()> {
    env_logger::init();
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
                writeln!(
                    file,
                    "{:?}",
                    result
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                )?;
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
                writeln!(
                    file,
                    "{:?}",
                    result
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                )?;
            }
        }
        Command::Sample { seed } => todo!(),
    };

    // match mode.as_str() {
    //     "test" => {
    //         let data = data::read_test(input)?;

    //         let result = optim::optimize_multivariable(&data)?;
    //         println!("{:?}", result);

    //         let mut file = std::fs::File::create(output)?;
    //         writeln!(file, "{} {}", result.0, result.1)?;
    //     }
    //     "optim" => {
    //         let data = data::read_divergences(input)?;
    //         let result = optim::optimize(&data)?;
    //         println!("{:?}", result);

    //         let mut file = std::fs::File::create(output)?;
    //         writeln!(file, "{:?}", result)?;
    //     }
    //     "boot" => {
    //         let rep: u64 = args[4].parse()?;
    //         let data = data::bootstrap_divergences(input, Some(rep))?;
    //         let result = optim::optimize(&data)?;
    //         println!("{:?}", result);

    //         let mut file = std::fs::File::create(output)?;
    //         writeln!(file, "{:?}", result)?;
    //     }

    //     "varboot" => {
    //         let rep: u64 = args[4].parse()?;
    //         let rm: f64 = args[5].parse()?;
    //         let om: f64 = args[6].parse()?;

    //         let data = data::bootstrap_divergences(input, Some(rep))?;

    //         let result = optim::optimize_variable(&data, rm, om)?;

    //         println!("{:?}", result);

    //         let mut file = std::fs::File::create(output)?;
    //         writeln!(file, "{:?}", result)?;
    //     }

    //     "multiboot" => {
    //         let rep: u64 = args[4].parse()?;
    //         let data = data::bootstrap_divergences(input, Some(rep))?;

    //         let result = optim::optimize_multivariable(&data)?;
    //         println!("{:?}", result);

    //         let mut file = std::fs::File::create(output)?;
    //         writeln!(file, "{} {}", result.0, result.1)?;
    //     }

    //     _ => panic!("unsupported mode!"),
    // }

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
    },
}
