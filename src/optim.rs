use crate::{
    data::SegmentDivergence,
    observation::Observation,
    parameter::{get_tuples_sub, Parameters},
};
use color_eyre::eyre::{bail, Result};
use ndarray::prelude::*;
use rayon::prelude::*;
use scirs2_optimize::{
    minimize_scalar,
    unconstrained::{minimize_powell, Bounds, Options},
};

pub fn optimize(data: &[SegmentDivergence], parameters: Parameters) -> Result<f64> {
    let obs: Vec<Observation> = data
        .iter()
        .map(|s| {
            Observation::new(
                s.k,
                s.mu,
                &parameters.c,
                &parameters.t,
                parameters.adm_f,
                parameters.adm_idx,
            )
        })
        .collect();

    let result = if parameters.c.num_fit() == 1 {
        // run optimization
        let ans = minimize_scalar(
            |val| {
                let param_tuples = get_tuples_sub(
                    &parameters.c,
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
        )?;

        log::debug!("{:?}", ans);

        // we are fitting n, so convert back from coalrate
        parameters.n1 / ans.x
    } else if parameters.t.num_fit() == 1 {
        let ans = minimize_scalar(
            |val| {
                let param_tuples = get_tuples_sub(
                    &parameters.c,
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
        )?;

        log::debug!("{:?}", ans);

        // we are fitting t, so convert back
        ans.x * 2. * parameters.n1
    } else {
        bail!("cannot perform single-variable optimization. check inputs!");
    };

    Ok(result)
}

pub fn optimize_multivariable(
    data: &[SegmentDivergence],
    parameters: Parameters,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let cv = parameters.c.num_fit();
    let tv = parameters.t.num_fit();

    let options = Options {
        bounds: Some(Bounds::from_vecs(
            vec![Some(1.0); cv + tv],
            vec![Some(100000.0); cv + tv],
        )?),
        ..Default::default()
    };

    let obs: Vec<Observation> = data
        .iter()
        .map(|s| {
            Observation::new(
                s.k,
                s.mu,
                &parameters.c,
                &parameters.t,
                parameters.adm_f,
                parameters.adm_idx,
            )
        })
        .collect();

    let result = minimize_powell(
        |fit_vals| {
            let fit_vals = fit_vals.as_slice().unwrap();

            let param_tuples = get_tuples_sub(
                &parameters.c,
                &parameters.t,
                &fit_vals[0..cv],
                &fit_vals[cv..(cv + tv)],
            );

            let total: f64 = obs.par_iter().map(|o| o.lpdf(&param_tuples)).sum();

            -total
        },
        Array1::from_vec([parameters.c.fit(), parameters.t.fit()].concat()),
        &options,
    )?;

    log::debug!("{:?}", result);

    let n_ans: Vec<f64> = result.x.to_vec()[0..cv]
        .iter()
        .map(|x| parameters.n1 / x)
        .collect();
    let t_ans: Vec<f64> = result.x.to_vec()[cv..(cv + tv)]
        .iter()
        .map(|x| x * 2. * parameters.n1)
        .collect();

    Ok((n_ans, t_ans))
}
