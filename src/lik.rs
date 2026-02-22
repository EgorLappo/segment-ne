pub fn log_intergral_exact(k: f64, segment_start: f64, segment_end: f64, n: f64, mu: f64) -> f64 {
    if segment_start == 0.0 {
        let mut ans = k * (2. * mu * segment_end).ln();
        ans += xsf::gammainc(1. + k, (2. * mu + 0.5 / n) * segment_end).ln();
        ans -= (1. + 4. * mu * n).ln();
        ans -= k * ((2. * mu + 0.5 / n) * segment_end).ln();

        return ans;
    }

    let mut ans = xsf::gammaincc(1. + k, (2. * mu + 0.5 / n) * segment_start)
        - xsf::gammaincc(1. + k, (2. * mu + 0.5 / n) * segment_end);

    ans = ans.ln() - (1. + k) * (2. * mu + 0.5 / n).ln()
        + k * mu.ln()
        + k * (2.0_f64).ln()
        + segment_start / (2. * n)
        - (2. * n).ln();

    ans
}

pub fn log_integral_exact_inf(k: f64, segment_start: f64, n: f64, mu: f64) -> f64 {
    if segment_start == 0.0 {
        let mut ans = k * (2. * mu).ln() - (2. * n).ln();
        ans -= (k + 1.) * (2. * mu + 1. / (2. * n)).ln();
        return ans;
    }

    log_neg_antiderivative(k, segment_start, segment_start, n, mu)
}

pub fn log_neg_antiderivative(k: f64, t: f64, segment_start: f64, n: f64, mu: f64) -> f64 {
    let mut ans = (segment_start / 2. / n)
        + t.ln()
        + k * (2. * mu * t).ln()
        + xsf::gammaincc(1. + k, (2. * mu + 0.5 / n) * t).ln();

    ans = ans - (k + 1.) * ((2. * mu + 0.5 / n) * t).ln() - (2. * n).ln();

    ans
}

pub fn log_lognormal_pdf(x: f64, mu: f64, sd: f64) -> f64 {
    -(x * sd).ln() - (x.ln() - mu.ln()).powi(2) / (2. * sd * sd)
}
