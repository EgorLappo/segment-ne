use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rand_distr::StandardNormal;
use rayon::prelude::*;
use std::collections::VecDeque;

use crate::data::SegmentDivergence;
use crate::observation::Observation;
use crate::parameter::{ParameterList, Parameters, get_tuples, get_tuples_sub};

const ACC_RATE_LO: usize = 25;
const ACC_RATE_HI: usize = 35;
const SD_INIT: f64 = 10.;
const SD_UPDATE_RATE: f64 = 0.05;
const N_RECENT_STEPS: usize = 100;

#[derive(Debug, Clone)]
pub struct Chain {
    pub n: ParameterList,
    pub t: ParameterList,
    pub obs: Vec<Observation>,
    pub loglik: f64,
    sd: f64, // std. dev. of the proposal
    step_count: usize,
    // store recent
    steps: VecDeque<u8>,
}

type ChainOutput = (Vec<f64>, Vec<Box<[f64]>>, Vec<Box<[f64]>>);

impl Chain {
    pub fn new(data: &[SegmentDivergence], parameters: Parameters) -> Self {
        let n = parameters.n.clone();
        let t = parameters.t.clone();

        let obs: Vec<Observation> = data
            .iter()
            .map(|s| Observation::new(s.k, s.mu, &n, &t, parameters.adm_p, parameters.adm_idx))
            .collect();

        let param_tuples = get_tuples(&n, &t);

        let loglik = obs.iter().map(|o| o.lpdf(&param_tuples)).sum();

        let steps = vec![0; N_RECENT_STEPS].into();

        Self {
            n,
            t,
            obs,
            loglik,
            sd: SD_INIT,
            step_count: 0,
            steps,
        }
    }

    fn step<R: Rng>(&mut self, rng: &mut R) {
        // propose
        let new_nfit: Vec<f64> = self
            .n
            .fit()
            .iter()
            .zip(rng.sample_iter::<f64, _>(StandardNormal))
            .map(|(n, x)| n + self.sd * x)
            .collect();

        // TODO: note that this is ill-defined right now
        // because t has to be ordered
        let new_tfit: Vec<f64> = self
            .t
            .fit()
            .iter()
            .zip(rng.sample_iter::<f64, _>(StandardNormal))
            .map(|(t, x)| t + self.sd * x)
            .collect();

        let new_param_tuples = get_tuples_sub(&self.n, &self.t, &new_nfit, &new_tfit);

        // hack: return/reject immediately if times not ordered
        if !new_param_tuples
            .iter()
            .map(|x| x.0.0.unwrap())
            .is_sorted_by(|a, b| a < b)
        {
            log::warn!("t proposal: not ordered, rejecting right away");
            self.step_count += 1;
            self.steps.pop_front();
            self.steps.push_back(0);
        }

        let new_loglik = self.obs.par_iter().map(|o| o.lpdf(&new_param_tuples)).sum();

        // NOTE to future self:
        //  we use *symmetric* gaussian proposal
        //  so we don't have to add proposal distribution density terms here
        let log_ratio: f64 = new_loglik - self.loglik;

        log::debug!(
            "step {}: n: {:?} -> {:?}",
            self.step_count,
            self.n.fit(),
            new_nfit
        );
        log::debug!(
            "step {}: t: {:?} -> {:?}",
            self.step_count,
            self.t.fit(),
            new_tfit
        );
        log::debug!(
            "step {}: ll: {} -> {}",
            self.step_count,
            self.loglik,
            new_loglik
        );

        if rand::random::<f64>() <= log_ratio.exp() {
            // accept
            self.n.set_fit(&new_nfit);
            self.t.set_fit(&new_tfit);

            self.loglik = new_loglik;

            // log acceptance
            self.steps.pop_front();
            self.steps.push_back(1);

            log::debug!("step {}: accepting", self.step_count);
        } else {
            // reject
            self.steps.pop_front();
            self.steps.push_back(0);

            log::debug!("step {}: rejecting", self.step_count);
        }

        self.step_count += 1;

        // update acceptance rate to be ~30%
        if (self.step_count > 500) && self.step_count.is_multiple_of(100) {
            let acc_rate = self.steps.iter().sum::<u8>() as usize;
            log::debug!(
                "step {}: acceptance rate {:.02}, sd {:.02}",
                self.step_count,
                acc_rate,
                self.sd
            );
            if acc_rate > ACC_RATE_HI {
                self.sd *= 1. + SD_UPDATE_RATE;
            } else if acc_rate < ACC_RATE_LO {
                self.sd *= 1. - SD_UPDATE_RATE;
            }

            self.sd = self.sd.min(100.);
            self.sd = self.sd.max(5.);
        }
    }

    pub fn run(
        &mut self,
        warmup: usize,
        sampling: usize,
        seed: u64,
        bar: MultiProgress,
    ) -> ChainOutput {
        let mut lls = Vec::with_capacity(sampling);
        let mut n_samples = Vec::with_capacity(sampling);
        let mut t_samples = Vec::with_capacity(sampling);

        let mut rng = SmallRng::seed_from_u64(seed);

        let pb = bar.add(ProgressBar::new((warmup + sampling) as u64));
        let style = ProgressStyle::with_template(
            "{spinner:.purple} {prefix}: [{elapsed}/{duration}] [{bar:.cyan/blue}] {human_pos}/{human_len}",
        )
        .unwrap();
        pb.set_style(style);

        pb.set_prefix("warmup");
        for _ in 0..warmup {
            self.step(&mut rng);

            pb.inc(1);
        }

        pb.set_prefix("sampling");
        for _ in 0..sampling {
            self.step(&mut rng);

            lls.push(self.loglik);
            n_samples.push(self.n.fit().into());
            t_samples.push(self.t.fit().into());

            pb.inc(1);
        }

        pb.finish();

        (lls, n_samples, t_samples)
    }
}

// fn get_loglik(obs: &[Observation], n: &[f64], t: &[f64]) -> f64 {
//     obs.iter().map(|o| o.lpdf(n, t)).sum()
// }

// outline
// 1. initialize to store data and parameters
//    format: fixed recent times, fixed ancient times, intermediate times of interest
//    store each observation in a struct such that we can re-compute its likelihood efficiently
// 2. proposals: adjust the population sizes (mv-normal distribution)
//               shift the change times (later, using mv-normal also)
//               decay the variance of proposal distributions?
// 3. sampling
