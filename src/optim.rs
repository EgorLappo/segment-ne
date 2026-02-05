use crate::{
    data::SegmentDivergence,
    observation::Observation,
    parameter::{Parameters, get_tuples_sub},
};
use color_eyre::eyre::{Result, bail};
use ndarray::prelude::*;
use rayon::prelude::*;
use scirs2_optimize::{
    minimize_scalar,
    unconstrained::{Bounds, Options, minimize_powell},
};

pub fn optimize(data: &[SegmentDivergence], parameters: Parameters) -> Result<f64> {
    let obs: Vec<Observation> = data
        .iter()
        .map(|s| Observation::new(s.k, s.mu, &parameters.n, &parameters.t))
        .collect();

    let result = if parameters.n.num_fit() == 1 {
        minimize_scalar(
            |val| {
                let param_tuples = get_tuples_sub(
                    &parameters.n,
                    &parameters.t,
                    std::slice::from_ref(&val),
                    &Vec::new(),
                );

                let total: f64 = obs.par_iter().map(|o| o.lpdf(&param_tuples)).sum();

                -total
            },
            Some((1.0, 100000.0)),
            scirs2_optimize::scalar::Method::Bounded,
            None,
        )?
    } else if parameters.t.num_fit() == 1 {
        minimize_scalar(
            |val| {
                let param_tuples = get_tuples_sub(
                    &parameters.n,
                    &parameters.t,
                    &Vec::new(),
                    std::slice::from_ref(&val),
                );

                let total: f64 = obs.par_iter().map(|o| o.lpdf(&param_tuples)).sum();

                -total
            },
            Some((1.0, 100000.0)),
            scirs2_optimize::scalar::Method::Bounded,
            None,
        )?
    } else {
        bail!("cannot perform single-variable optimization. check inputs!");
    };

    log::debug!("{:?}", result);

    Ok(result.x)
}

pub fn optimize_multivariable(
    data: &[SegmentDivergence],
    parameters: Parameters,
) -> Result<Vec<f64>> {
    let nv = parameters.n.num_fit();
    let tv = parameters.t.num_fit();

    let options = Options {
        bounds: Some(Bounds::from_vecs(
            vec![Some(1.0); nv + tv],
            vec![Some(100000.0); nv + tv],
        )?),
        ..Default::default()
    };

    let obs: Vec<Observation> = data
        .iter()
        .map(|s| Observation::new(s.k, s.mu, &parameters.n, &parameters.t))
        .collect();

    let result = minimize_powell(
        |fit_vals| {
            let fit_vals = fit_vals.as_slice().unwrap();

            let param_tuples = get_tuples_sub(
                &parameters.n,
                &parameters.t,
                &fit_vals[0..nv],
                &fit_vals[nv..(nv + tv)],
            );

            let total: f64 = obs.par_iter().map(|o| o.lpdf(&param_tuples)).sum();

            -total
        },
        Array1::from_vec([parameters.n.init_values(), parameters.t.init_values()].concat()),
        &options,
    )?;

    log::debug!("{:?}", result);

    Ok(result.x.to_vec())
}
