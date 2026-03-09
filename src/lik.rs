pub fn log_integral_exact(k: f64, ts: f64, te: f64, c: f64, theta: f64) -> f64 {
    if ts == 0.0 {
        xsf::gammainc(1. + k, (c + theta) * te).ln() - (1. + k) * (c + theta).ln()
    } else {
        let numerator =
            xsf::gammaincc(1. + k, (c + theta) * ts) - xsf::gammaincc(1. + k, (c + theta) * te);

        numerator.ln() + c * ts - (1. + k) * (c + theta).ln()
    }
}

pub fn log_integral_exact_inf(k: f64, ts: f64, c: f64, theta: f64) -> f64 {
    if ts == 0.0 {
        -(1. + k) * (c + theta).ln()
    } else {
        c * ts + xsf::gammaincc(1. + k, (c + theta) * ts).ln() - (1. + k) * (c + theta).ln()
    }
}
