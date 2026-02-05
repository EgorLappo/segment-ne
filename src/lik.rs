use logsumexp::LogSumExp;

pub fn k_lpdf(k: f64, pop_sizes: &[f64], change_times: &[f64], mu: f64) -> f64 {
    let s = pop_sizes.len();

    let mut ans: f64 = 0.0;
    let mut total: Vec<f64> = Vec::with_capacity(s);

    for ((&segment_start, &segment_end), &pop_size) in change_times
        .iter()
        .zip(change_times.iter().skip(1))
        .zip(pop_sizes.iter())
    {
        let segment_length = segment_end - segment_start;

        let term = log_intergral_exact(k, segment_start, segment_end, pop_size, mu);
        total.push(term + ans);

        ans += -segment_length / (2. * pop_size);
    }

    let segment_start = change_times[change_times.len() - 1];
    let pop_size = pop_sizes[pop_sizes.len() - 1];

    let term = log_integral_exact_inf(k, segment_start, pop_size, mu);
    total.push(term + ans);

    total.iter().ln_sum_exp()
}

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
