use rand::{rngs::SmallRng, Rng, SeedableRng};
use rand_distr::StandardNormal;

use crate::data::SegmentDivergence;
use crate::parameter::{ParameterList, Parameters};

#[derive(Debug, Clone)]
pub struct Chain {
    pub n: ParameterList,
    pub t: ParameterList,
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
    // term cache goes here
}

impl Observation {
    pub fn new(k: f64, mu: f64) -> Self {
        Self { k, mu }
    }

    pub fn lpdf(&self, n: &[f64], t: &[f64]) -> f64 {
        // draft: use full computation each time
        crate::lik::k_lpdf(self.k, n, t, self.mu)
    }
}

impl Chain {
    pub fn new(data: &[SegmentDivergence], parameters: Parameters) -> Self {
        // split pop sizes
        let n = parameters.n.clone();
        let t = parameters.t.clone();

        let obs: Vec<Observation> = data.iter().map(|s| Observation::new(s.k, s.mu)).collect();

        let loglik = get_loglik(&obs, &n.vec(), &t.vec());

        Self {
            n,
            t,
            obs,
            loglik,
            sd: 100.0,
            step_count: 0.0,
            accept_count: 0.0,
        }
    }

    fn step<R: Rng>(&mut self, rng: &mut R) {
        // propose
        let new_nfit: Vec<f64> = self
            .n
            .fit()
            .iter()
            .zip(rng.sample_iter::<f64, StandardNormal>(StandardNormal))
            .map(|(n, x)| n + self.sd * x)
            .collect();

        let new_n = self.n.substitute(&new_nfit);
        // don't change for now
        let new_t = self.t.vec();

        let new_loglik = get_loglik(&self.obs, &new_n, &new_t);

        let log_ratio = new_loglik - self.loglik;

        log::info!(
            "step: new params {:?}; log_ratio {:.02}",
            new_nfit,
            log_ratio
        );

        if rand::random::<f64>() <= log_ratio.exp() {
            // accept
            self.n.set_fit(&new_nfit);
            // dont touch t
            self.loglik = new_loglik;
            self.accept_count += 1.;
        } else {
            // reject
        }

        self.step_count += 1.;
    }

    pub fn run(&mut self, warmup: usize, sampling: usize, seed: u64) -> Vec<(f64, Box<[f64]>)> {
        let mut samples = Vec::with_capacity(sampling);
        let mut rng = SmallRng::seed_from_u64(seed);

        for _ in 0..warmup {
            self.step(&mut rng);
        }

        for _ in 0..sampling {
            self.step(&mut rng);
            samples.push((self.loglik, self.n.fit().into()));
        }

        samples
    }
}

fn get_loglik(obs: &[Observation], n: &[f64], t: &[f64]) -> f64 {
    obs.iter().map(|o| o.lpdf(n, t)).sum()
}

// outline
// 1. initialize to store data and parameters
//    format: fixed recent times, fixed ancient times, intermediate times of interest
//    store each observation in a struct such that we can re-compute its likelihood efficiently
// 2. proposals: adjust the population sizes (mv-normal distribution)
//               shift the change times (later, using mv-normal also)
//               decay the variance of proposal distributions?
// 3. sampling
