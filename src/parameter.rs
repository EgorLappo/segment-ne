use color_eyre::eyre::{Result, WrapErr, bail};

#[derive(Debug, Clone)]
enum ParameterValue {
    Fit(f64),
    Fixed(f64),
}

#[derive(Debug, Clone)]
pub struct ParameterList {
    rec: Box<[f64]>,
    fit: Box<[f64]>,
    anc: Box<[f64]>,
}

impl ParameterList {
    pub fn new(rec: &[f64], fit: &[f64], anc: &[f64]) -> Self {
        Self {
            rec: rec.into(),
            fit: fit.into(),
            anc: anc.into(),
        }
    }

    pub fn init_values(&self) -> &[f64] {
        &self.fit
    }

    pub fn len(&self) -> usize {
        self.rec.len() + self.fit.len() + self.anc.len()
    }

    pub fn num_fit(&self) -> usize {
        self.fit.len()
    }

    pub fn fit(&self) -> &[f64] {
        &self.fit
    }

    pub fn set_fit(&mut self, val: &[f64]) {
        self.fit = val.into();
    }

    // helpers for "uncached computation"
    pub fn vec(&self) -> Box<[f64]> {
        self.rec
            .iter()
            .copied()
            .chain(self.fit.iter().copied())
            .chain(self.anc.iter().copied())
            .collect()
    }

    pub fn substitute(&self, sub: &[f64]) -> Vec<f64> {
        itertools::chain!(self.rec.iter(), sub.iter(), self.anc.iter())
            .copied()
            .collect()
    }

    // helpers for cached computation
    pub fn fit_idx(&self) -> Vec<usize> {
        let init = self.rec.len();
        (init..(init + self.fit.len())).collect()
    }
}

#[derive(Debug, Clone)]
pub struct Parameters {
    pub n: ParameterList,
    pub t: ParameterList,
}

impl Parameters {
    pub fn new(size_str: &str, time_str: &str) -> Result<Self> {
        let pop_sizes = parse_params(size_str)?;
        let mut change_times = parse_params(time_str)?;

        // validate parameter lengths
        if change_times.len() + 1 == pop_sizes.len() {
            log::warn!(
                "number of time parameters is one less than the number of size parameters. assuming you did not include time zero"
            );
            change_times.insert(0, ParameterValue::Fixed(0.));
        } else if change_times.len() != pop_sizes.len() {
            bail!(
                "provided {:?} sizes but {:?} times",
                pop_sizes.len(),
                change_times.len()
            );
        }

        let (n_rec, n_fit, n_anc) = split_params(&pop_sizes);
        let (t_rec, t_fit, t_anc) = split_params(&change_times);

        Ok(Self {
            n: ParameterList::new(&n_rec, &n_fit, &n_anc),
            t: ParameterList::new(&t_rec, &t_fit, &t_anc),
        })
    }
}

// HELPER FUNCTIONS

fn parse_params(param_str: &str) -> Result<Vec<ParameterValue>> {
    param_str
        .split_ascii_whitespace()
        .map(|x| {
            let param = if x.starts_with('~') {
                if x.len() == 1 {
                    bail!(
                        "please provide initial values for all parameters: {}",
                        param_str
                    );
                } else {
                    let num = x[1..x.len()].parse()?;
                    ParameterValue::Fit(num)
                }
            } else {
                let num = x
                    .parse()
                    .wrap_err_with(|| format!("cannot parse parameter: {}", x))?;
                ParameterValue::Fixed(num)
            };
            Ok(param)
        })
        .collect::<Result<Vec<ParameterValue>>>()
}

/// Split the vector of Parameter enums into three parts:
/// - recent fixed  (f64)
/// - possibly initialized variable (Option<f64>)
/// - ancient fixed (f64)
fn split_params(params: &[ParameterValue]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut values_r = Vec::new();
    let mut values_fit = Vec::new();
    let mut values_a = Vec::new();

    let mut is_recent = true;

    for p in params.iter() {
        match p {
            ParameterValue::Fit(v) => {
                if is_recent {
                    is_recent = false;
                }

                values_fit.push(*v);
            }
            ParameterValue::Fixed(v) => {
                if is_recent {
                    values_r.push(*v);
                } else {
                    values_a.push(*v);
                }
            }
        }
    }

    (values_r, values_fit, values_a)
}
