use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn compute_lpdf(c: &mut Criterion) {
    return;
}

criterion_group!(lpdf, compute_lpdf);
criterion_main!(lpdf);
