use crate::{lik, parameter::ParamTuples};
use logsumexp::LogSumExp;

use crate::parameter::{ParameterList, get_should_cache, get_tuples};

#[derive(Debug, Clone)]
pub struct Observation {
    pub k: f64,
    pub mu: f64,
    // term cache goes here
    term_cache: Vec<Option<f64>>,
}

impl Observation {
    pub fn new(k: f64, mu: f64, n: &ParameterList, t: &ParameterList) -> Self {
        // init cache
        let mut term_cache = Vec::new();

        let param_tuples = get_tuples(n, t);
        let should_cache = get_should_cache(n, t);

        for (((ot_start, ot_end), pop_size), do_cache) in
            param_tuples.iter().zip(should_cache.iter())
        {
            if *do_cache {
                match (&ot_start, &ot_end) {
                    (Some(segment_start), Some(segment_end)) => {
                        let term = lik::log_intergral_exact(
                            k,
                            *segment_start,
                            *segment_end,
                            *pop_size,
                            mu,
                        );

                        term_cache.push(Some(term));
                    }
                    (Some(segment_start), Option::None) => {
                        let term = lik::log_integral_exact_inf(k, *segment_start, *pop_size, mu);
                        term_cache.push(Some(term));
                    }
                    _ => unreachable!(),
                }
            } else {
                term_cache.push(None);
            }
        }

        Self { k, mu, term_cache }
    }

    pub fn lpdf(&self, p: &ParamTuples) -> f64 {
        // draft: use full computation each time

        let mut ans: f64 = 0.0;
        let mut total: Vec<f64> = Vec::with_capacity(10);

        for (((ot_start, ot_end), pop_size), cache) in p.iter().zip(self.term_cache.iter()) {
            match (&ot_start, &ot_end) {
                (Some(segment_start), Some(segment_end)) => {
                    let segment_length = segment_end - segment_start;
                    let term = cache.unwrap_or_else(|| {
                        lik::log_intergral_exact(
                            self.k,
                            *segment_start,
                            *segment_end,
                            *pop_size,
                            self.mu,
                        )
                    });

                    total.push(term + ans);

                    ans += -segment_length / (2. * pop_size);
                }
                // NOTE: rust-analyzer bug? Doesn't see this None as enum variant
                (Some(segment_start), Option::None) => {
                    let term = cache.unwrap_or_else(|| {
                        lik::log_integral_exact_inf(self.k, *segment_start, *pop_size, self.mu)
                    });
                    total.push(term + ans);
                }
                _ => unreachable!(),
            }
        }

        total.iter().ln_sum_exp()
    }
}
