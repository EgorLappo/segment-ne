use crate::{data::SegmentDivergence, lik, parameter::Parameters};
use color_eyre::eyre::Result;
use ndarray::prelude::*;
use scirs2_optimize::{
    minimize_scalar,
    unconstrained::{Bounds, Options, minimize_powell},
};

pub fn optimize(data: &[SegmentDivergence], parameters: Parameters) -> Result<f64> {
    let ts = parameters.t.vec();
    let result = minimize_scalar(
        |n| {
            let ns = parameters.n.substitute(std::slice::from_ref(&n));

            let mut total = 0.0;

            for sd in data.iter() {
                total -= lik::k_lpdf(sd.k, &ns, &ts, sd.mu);
            }

            total
        },
        Some((1.0, 100000.0)),
        scirs2_optimize::scalar::Method::Bounded,
        None,
    )?;

    Ok(result.x)
}

pub fn optimize_multivariable(
    data: &[SegmentDivergence],
    parameters: Parameters,
) -> Result<Vec<f64>> {
    let nv = parameters.n.num_fit();

    let options = Options {
        bounds: Some(Bounds::from_vecs(
            vec![Some(1.0); nv],
            vec![Some(100000.0); nv],
        )?),
        ..Default::default()
    };

    let ts = parameters.t.vec();

    let result = minimize_powell(
        |n_fits| {
            let ns = parameters.n.substitute(n_fits.as_slice().unwrap());

            let mut total = 0.0;

            for sd in data.iter() {
                total -= lik::k_lpdf(sd.k, &ns, &ts, sd.mu);
            }

            total
        },
        Array1::from_vec(parameters.n.init_values().into()),
        &options,
    )?;

    Ok(result.x.to_vec())
}
