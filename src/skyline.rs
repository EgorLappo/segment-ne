use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rand::{rngs::SmallRng, Rng, RngExt, SeedableRng};
use rand_distr::multi::{Dirichlet, MultiDistribution};
use rand_distr::StandardNormal;
use rayon::prelude::*;
use std::collections::VecDeque;

use crate::data::SegmentDivergence;
use crate::observation::Observation;
use crate::parameter::{get_tuples, get_tuples_sub, ParameterList, Parameters};

const ACC_RATE_LO: usize = 25;
const ACC_RATE_HI: usize = 35;
const SD_INIT: f64 = 0.1;
const SD_MIN: f64 = 0.;
const SD_MAX: f64 = 3.;
const SD_UPDATE_RATE: f64 = 0.01;
const N_RECENT_STEPS: usize = 100;

// laplace prior for difference of population sizes
const N_PRIOR_B: f64 = 1.;

#[derive(Debug, Clone)]
pub struct SkylineChain {
    n_scale: f64,
    pub log_c: ParameterList,
    pub t: ParameterList,
    pub obs: Vec<Observation>,
    pub loglik: f64,
    t_prior: DirichletPrior,
    t_update_every: usize,
    sd: f64, // std. dev. of the proposal
    step_count: usize,
    // store recent
    steps: VecDeque<u8>,
}

#[derive(Debug, Clone)]
struct DirichletPrior {
    // ntr: usize,
    // scale: f64,
    // alpha: f64,
    // lambdas: Vec<f64>,
    dist: Dirichlet<f64>,
    target_offset: f64,
    target_mult: f64,
    values: Vec<f64>,
}

impl DirichletPrior {
    fn new(ntr: usize, scale: f64, alpha: f64, target_interval: (f64, f64)) -> Self {
        let mut lambdas = vec![0.0; ntr];

        lambdas[0] = scale.powi(1 - ntr as i32);
        for (i, l) in lambdas.iter_mut().enumerate().skip(1) {
            *l = (scale - 1.) * scale.powi(i as i32 - ntr as i32);
        }

        let dist =
            Dirichlet::new(&lambdas.iter().map(|x| x * alpha).collect::<Vec<f64>>()).unwrap();

        // fill the value vector with expected values to begin with
        let values = lambdas.clone();

        Self {
            // ntr,
            // scale,
            // alpha,
            // lambdas,
            dist,
            target_offset: target_interval.0,
            target_mult: target_interval.1 - target_interval.0,
            values,
        }
    }

    fn sample<R: Rng>(&mut self, rng: &mut R) -> Vec<f64> {
        self.dist.sample_to_slice(rng, &mut self.values);
        self.values
            .iter()
            // now compute cumulative sum and truncate (last entry will be 1)
            .scan(0.0, |acc, x| {
                *acc += x;
                if *acc >= 1.0 {
                    None
                } else {
                    Some(*acc)
                }
            })
            // map values to target interval
            .map(|x| self.target_offset + x * self.target_mult)
            .collect()
    }
}

type ChainOutput = (Vec<f64>, Vec<Box<[f64]>>, Vec<Box<[f64]>>, Vec<Box<[f64]>>);

impl SkylineChain {
    pub fn new(
        data: &[SegmentDivergence],
        parameters: Parameters,
        num_intervals: usize,
        t_scale: f64,
        alpha: f64,
    ) -> Self {
        let n_scale = parameters.n_scale;
        let log_c = parameters.log_c.clone();
        let mut t = parameters.t.clone();

        // initialize the prior for t
        let t_interval = t.bounds_unchecked();
        let t_prior = DirichletPrior::new(num_intervals, t_scale, alpha, t_interval);

        log::debug!("got following lambdas for t prior: {:?}", t_prior.values);

        // fill out the values with expected values (so that we don't have to pass Rng to new())
        let new_t: Vec<f64> = t_prior
            // values contain lambdas right after init
            .values
            .iter()
            // now compute cumulative sum and truncate (last entry will be 1)
            .scan(0.0, |acc, x| {
                *acc += x;
                if *acc >= 1.0 {
                    None
                } else {
                    Some(*acc)
                }
            })
            // map values to target interval
            .map(|x| t_interval.0 + x * (t_interval.1 - t_interval.0))
            .collect();
        log::debug!("initializing t with these values: {:?}", new_t);

        t.set_fit(&new_t);

        let obs: Vec<Observation> = data
            .iter()
            .map(|s| {
                // we get L*mu_bp from the data
                // we want to fit with theta = 4 N_1 mu
                let theta = 4. * s.mu * parameters.n_scale;
                Observation::new(s.k, theta, &log_c, &t, parameters.adm_f, parameters.adm_idx)
            })
            .collect();

        let param_tuples = get_tuples(&log_c, &t);

        let loglik =
            obs.iter().map(|o| o.lpdf(&param_tuples)).sum::<f64>() + c_log_prior(log_c.fit());

        let steps = vec![0; N_RECENT_STEPS].into();

        // NOTE: expose this option later?
        // step switch: how often to change t vs n (do it on different steps)
        let t_update_every = 2;

        Self {
            n_scale,
            log_c,
            t,
            obs,
            loglik,
            t_prior,
            t_update_every,
            sd: SD_INIT,
            step_count: 0,
            steps,
        }
    }

    fn step<R: Rng>(&mut self, rng: &mut R) {
        // if `t_update_every` is 1, update both coalrates and t on each step
        // otherwise, update t every `t_update_every`)))
        let new_lcfit: Vec<f64> =
            if self.t_update_every == 1 || !self.step_count.is_multiple_of(self.t_update_every) {
                // propose
                self.log_c
                    .fit()
                    .iter()
                    .zip(rng.sample_iter::<f64, _>(StandardNormal))
                    .map(|(lc, x)| lc + self.sd * x)
                    .collect()
            } else {
                self.log_c.fit().to_vec()
            };

        let new_tfit: Vec<f64> =
            if self.t_update_every == 1 || self.step_count.is_multiple_of(self.t_update_every) {
                self.t_prior.sample(rng)
            } else {
                self.t.fit().to_vec()
            };

        let new_param_tuples = get_tuples_sub(&self.log_c, &self.t, &new_lcfit, &new_tfit);

        let new_loglik = self
            .obs
            .par_iter()
            .map(|o| o.lpdf(&new_param_tuples))
            .sum::<f64>()
            + c_log_prior(&new_lcfit);

        // NOTE to future self:
        //  we use *symmetric* gaussian proposal
        //  so we don't have to add proposal distribution density terms here
        let log_ratio: f64 = new_loglik - self.loglik;

        log::debug!(
            "step {}: log-c: {:?} -> {:?}",
            self.step_count,
            self.log_c.fit(),
            new_lcfit
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
            self.log_c.set_fit(&new_lcfit);
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

            self.sd = self.sd.min(SD_MAX);
            self.sd = self.sd.max(SD_MIN);
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
        let mut log_c_samples = Vec::with_capacity(sampling);
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
            log_c_samples.push(self.log_c.fit().iter().cloned().collect());
            n_samples.push(
                self.log_c
                    .fit()
                    .iter()
                    .map(|x| self.n_scale / x.exp())
                    .collect(),
            );
            t_samples.push(self.t.fit().iter().map(|x| x * 2. * self.n_scale).collect());

            pb.inc(1);
        }

        pb.finish();

        (lls, n_samples, t_samples, log_c_samples)
    }
}

// helper functions

fn c_log_prior(ns: &[f64]) -> f64 {
    ns.iter()
        .zip(ns.iter().skip(1))
        // log laplace density
        .map(|(x, y)| -(x - y).abs() / N_PRIOR_B)
        .sum()
}
