use color_eyre::eyre::{bail, Result, WrapErr};
use itertools::Itertools;

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

    pub fn num_fit(&self) -> usize {
        self.fit.len()
    }

    pub fn fit(&self) -> &[f64] {
        &self.fit
    }

    pub fn set_fit(&mut self, val: &[f64]) {
        self.fit = val.into();
    }

    pub fn bounds_unchecked(&self) -> (f64, f64) {
        (*self.rec.last().unwrap(), *self.anc.first().unwrap())
    }
}

pub type ParamTuples = Box<[((Option<f64>, Option<f64>), f64)]>;

pub fn get_tuples(n: &ParameterList, t: &ParameterList) -> ParamTuples {
    let ti = itertools::chain!(t.rec.iter(), t.fit.iter(), t.anc.iter()).copied();
    let ni = itertools::chain!(n.rec.iter(), n.fit.iter(), n.anc.iter()).copied();

    ti.map(Some)
        .chain(std::iter::once(None))
        .tuple_windows::<(Option<f64>, Option<f64>)>()
        .zip(ni)
        .collect()
}

pub fn get_tuples_sub(
    n: &ParameterList,
    t: &ParameterList,
    n_sub: &[f64],
    t_sub: &[f64],
) -> ParamTuples {
    let ti = itertools::chain!(t.rec.iter(), t_sub.iter(), t.anc.iter()).copied();
    let ni = itertools::chain!(n.rec.iter(), n_sub.iter(), n.anc.iter()).copied();

    ti.map(Some)
        .chain(std::iter::once(None))
        .tuple_windows::<(Option<f64>, Option<f64>)>()
        .zip(ni)
        .collect()
}

pub fn get_should_cache(n: &ParameterList, t: &ParameterList) -> Vec<bool> {
    let ni = itertools::chain!(
        n.rec.iter().map(|_| false),
        n.fit.iter().map(|_| true),
        n.anc.iter().map(|_| false)
    );
    let ti = itertools::chain!(
        t.rec.iter().map(|_| false),
        t.fit.iter().map(|_| true),
        t.anc.iter().map(|_| false)
    );

    ti.map(Some)
        .chain(std::iter::once(None))
        .tuple_windows()
        .zip(ni)
        .map(|((x, y), z)| !(x.unwrap_or(false) || y.unwrap_or(false) || z))
        .collect()
}

#[derive(Debug, Clone)]
pub struct Parameters {
    pub n1: f64,          // fixed first size for scaling
    pub c: ParameterList, // coalescence rates
    pub t: ParameterList, // rate change times
    pub adm_f: f64,       // admixture fraction
    pub adm_idx: usize,   // admixture location
}

#[derive(Debug, Clone)]
enum ParameterValue {
    Fit(f64),
    Fixed(f64),
}

impl Parameters {
    pub fn new(
        size_str: &str,
        time_str: &str,
        admixture_fraction: f64,
        admixture_index: usize,
    ) -> Result<Self> {
        let pop_sizes = parse_params(size_str)?;
        let mut change_times = parse_params(time_str)?;

        // validate parameter lengths
        if pop_sizes.len() < 2 {
            bail!(
                "error in sizes {}: please provide at least 2 sizes",
                size_str
            );
        } else if change_times.len() + 1 == pop_sizes.len() {
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

        if admixture_fraction <= 0. || admixture_fraction > 1.0 {
            bail!(
                "provided invalid admixture fraction {}. it must lie in (0, 1]",
                admixture_fraction
            )
        }

        if admixture_index == 0 || admixture_index >= pop_sizes.len() {
            bail!(
                "provided invalid admixture index {}. it must lie between provided constant population size segments",
                admixture_index
            )
        }

        let (n_rec, n_fit, n_anc) = split_params(&pop_sizes);
        let (t_rec, t_fit, t_anc) = split_params(&change_times);

        // transform parameters to coalescent scaling
        let c_rec: Vec<f64> = n_rec.iter().map(|x| n_rec[0] / x).collect();
        let c_fit: Vec<f64> = n_fit.iter().map(|x| n_rec[0] / x).collect();
        let c_anc: Vec<f64> = n_anc.iter().map(|x| n_rec[0] / x).collect();
        let tc_rec: Vec<f64> = t_rec.iter().map(|x| x / 2. / n_rec[0]).collect();
        let tc_fit: Vec<f64> = t_fit.iter().map(|x| x / 2. / n_rec[0]).collect();
        let tc_anc: Vec<f64> = t_anc.iter().map(|x| x / 2. / n_rec[0]).collect();

        // for now, make sure that the first segment is known
        if n_rec.len() == 0 {
            bail!(
                "error in sizes {}: cannot treat the most recent population size as inferrable",
                size_str
            );
        }

        Ok(Self {
            n1: n_rec[0],
            c: ParameterList::new(&c_rec, &c_fit, &c_anc),
            t: ParameterList::new(&tc_rec, &tc_fit, &tc_anc),
            adm_f: admixture_fraction,
            // NOTE: let users have 1-based, and we will use 0-based
            adm_idx: admixture_index - 1,
        })
    }

    pub fn expand_skyline(self, r: usize) -> Result<Self> {
        // ok here there is a lot of messy code that essentially tries to make
        // a new method work without changing the notation

        // we must validate and transform parameters because now a single `~` marker on n actually corresponds to *many* parameters being inferred!

        // first make sure a single 'n' is marked to be inferred
        if !(self.c.num_fit() == 1 && self.t.num_fit() == 0) {
            bail!("too many parameters marked for inference. for skyline runs, please mark a *single* population size value");
        }

        // since there were no t to be fit, all values are in "recent" vector
        // we need to split it so that t.rec AND t_anc

        // now we must split this interval into r
        // it's trivial with n
        let mut ec = self.c.clone();
        ec.set_fit(&vec![self.c.fit[0]; r]);

        // but with t we need one less values
        let mut et = self.t.clone();

        et.rec = self.t.rec.iter().cloned().take(ec.rec.len() + 1).collect();
        et.anc = self.t.rec.iter().cloned().skip(ec.rec.len() + 1).collect();
        // (fill with a dummy value, we will replace it in chain init)
        et.set_fit(&vec![0.0; r - 1]);

        if et.rec.is_empty() || et.anc.is_empty() {
            bail!("invalid time specification. for skyline runs, only inference on finite runs is supported")
        }

        Ok(Self {
            n1: self.n1,
            c: ec,
            t: et,
            adm_f: self.adm_f,
            adm_idx: self.adm_idx,
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
