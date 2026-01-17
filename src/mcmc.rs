use std::rc::Rc;

use rand::{rngs::SmallRng, Rng, SeedableRng};
use rand_distr::StandardNormal;

use crate::data::SegmentDivergence;

#[derive(Debug, Clone)]
pub struct Chain {
    pub pop_sizes: Vec<f64>,
    pub change_times: Vec<f64>,
    pub obs: Vec<Observation>,
    pub loglik: f64,
    sd: f64, // std. dev. of the proposal
    step_count: f64,
    accept_count: f64,
}

#[derive(Debug, Clone)]
pub struct Observation {
    pub k: f64,
    pub mu: f64,
    pop_sizes_r: Rc<[f64]>, // add an Rc here for sharing
    change_times_r: Rc<[f64]>,
    pop_sizes_a: Rc<[f64]>,
    change_times_a: Rc<[f64]>,
    // term cache goes here
}

impl Observation {
    pub fn new(
        k: f64,
        mu: f64,
        pop_sizes_r: &Rc<[f64]>, // add an Rc here for sharing
        change_times_r: &Rc<[f64]>,
        pop_sizes_a: &Rc<[f64]>,
        change_times_a: &Rc<[f64]>,
    ) -> Self {
        Self {
            k,
            mu,
            pop_sizes_r: pop_sizes_r.clone(),
            change_times_r: change_times_r.clone(),
            pop_sizes_a: pop_sizes_a.clone(),
            change_times_a: change_times_a.clone(),
        }
    }

    fn param_iter<'a, 'b: 'a>(
        &'b self,
        pop_sizes: &'a [f64],
        change_times: &'a [f64],
    ) -> impl Iterator<Item = ((&'a f64, &'a f64), &'a f64)> {
        let size_iter = self
            .pop_sizes_r
            .iter()
            .chain(pop_sizes.iter())
            .chain(self.pop_sizes_a.iter());
        let time_iter = self
            .change_times_r
            .iter()
            .chain(change_times.iter())
            .chain(self.change_times_a.iter());
        time_iter.clone().zip(time_iter.skip(1)).zip(size_iter)
    }

    pub fn lpdf(&self, pop_sizes: &[f64], change_times: &[f64]) -> f64 {
        // draft: use full computation each time
        let sizes: Box<[f64]> = self
            .pop_sizes_r
            .iter()
            .chain(pop_sizes.iter())
            .chain(self.pop_sizes_a.iter())
            .copied()
            .collect();
        let times: Box<[f64]> = self
            .change_times_r
            .iter()
            .chain(change_times.iter())
            .chain(self.change_times_a.iter())
            .copied()
            .collect();
        crate::lik::k_lpdf(self.k, &sizes, &times, self.mu)
    }
}

impl Chain {
    pub fn new(
        data: &[SegmentDivergence],
        pop_sizes: &[f64],
        change_times: &[f64],
        fit_idx: usize,
        fit_num: usize,
    ) -> Self {
        // split pop sizes
        let pop_sizes_r = pop_sizes[0..fit_idx].to_vec().into();
        let pop_sizes_fit = pop_sizes[fit_idx..(fit_idx + fit_num)].to_vec();
        let pop_sizes_a = pop_sizes[(fit_idx + fit_num)..pop_sizes.len()]
            .to_vec()
            .into();

        let change_times_r = change_times[0..fit_idx].to_vec().into();
        let change_times_fit = change_times[fit_idx..(fit_idx + fit_num)].to_vec();
        let change_times_a = change_times[(fit_idx + fit_num)..change_times.len()]
            .to_vec()
            .into();

        let obs: Vec<Observation> = data
            .iter()
            .map(|s| {
                Observation::new(
                    s.k,
                    s.mu,
                    &pop_sizes_r,
                    &change_times_r,
                    &pop_sizes_a,
                    &change_times_a,
                )
            })
            .collect();

        let loglik = get_loglik(&obs, &pop_sizes_fit, &change_times_fit);

        Self {
            pop_sizes: pop_sizes_fit,
            change_times: change_times_fit,
            obs,
            loglik,
            sd: 100.0,
            step_count: 0.0,
            accept_count: 0.0,
        }
    }

    fn step<R: Rng>(&mut self, rng: &mut R) {
        // propose
        let new_sizes: Vec<f64> = self
            .pop_sizes
            .iter()
            .zip(rng.sample_iter::<f64, StandardNormal>(StandardNormal))
            .map(|(n, x)| n + self.sd * x)
            .collect();
        // don't change for now
        let new_times = self.change_times.clone();

        let new_loglik = get_loglik(&self.obs, &new_sizes, &new_times);

        let ratio = new_loglik / self.loglik;

        if rand::random::<f64>() <= ratio {
            // accept
            self.pop_sizes = new_sizes;
            self.change_times = new_times;
            self.loglik = new_loglik;
            self.accept_count += 1.;
        } else {
            // reject
        }

        self.step_count += 1.;
    }

    pub fn run(
        &mut self,
        n_burnin: usize,
        n_sample: usize,
        seed: u64,
    ) -> Vec<(f64, Vec<f64>, Vec<f64>)> {
        let mut samples = Vec::with_capacity(n_sample);
        let mut rng = SmallRng::seed_from_u64(seed);

        for _ in 0..n_burnin {
            self.step(&mut rng);
        }

        for _ in 0..n_sample {
            self.step(&mut rng);
            samples.push((
                self.loglik,
                self.pop_sizes.clone(),
                self.change_times.clone(),
            ));
        }

        samples
    }
}

// outline
// 1. initialize to store data and parameters
//    format: fixed recent times, fixed ancient times, intermediate times of interest
//    store each observation in a struct such that we can re-compute its likelihood efficiently
// 2. proposals: adjust the population sizes (mv-normal distribution)
//               shift the change times (later, using mv-normal also)
//               decay the variance of proposal distributions?
// 3. sampling

fn get_loglik(obs: &[Observation], pop_sizes: &[f64], change_times: &[f64]) -> f64 {
    obs.iter().map(|o| o.lpdf(pop_sizes, change_times)).sum()
}
