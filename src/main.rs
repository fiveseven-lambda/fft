use num::complex::Complex64;
use rand::prelude::*;
use rustfft::FftPlanner;

fn main() {
    let n = 3;
    let len = 1 << n;

    let mut rng = rand::thread_rng();
    let mut rustfft: Vec<_> = (0..len)
        .map(|_| Complex64::new(rng.gen(), rng.gen()))
        .collect();
    let mut my_fft = rustfft.clone();

    for x in &rustfft {
        println!("{}", x);
    }

    let mut planner = FftPlanner::<f64>::new();
    let plan = planner.plan_fft_forward(len);
    plan.process(&mut rustfft);

    fft(&mut my_fft, n);

    for (x, y) in rustfft.iter().zip(&my_fft) {
        println!("{} {}", x, y);
    }
}

fn fft(v: &mut Vec<Complex64>, n: u32) {
    let len = v.len();
    assert_eq!(len, 1 << n);
    let mask1 = len - 1;
    let zeta = zeta_table(len);
    for i in 0..n {
        let mask2 = mask1 >> i + 1;
        *v = (0..v.len())
            .map(|j| {
                let lower = j & mask2;
                let upper = j ^ lower;
                let shift = upper << 1 & mask1;
                v[shift | lower] + zeta[upper] * v[shift | mask2 + 1 | lower]
            })
            .collect();
    }
}

// 1 の len 乗根（ len 個）の配列を返す
fn zeta_table(len: usize) -> Vec<Complex64> {
    let mut ret = Vec::with_capacity(len);
    let pri = Complex64::from_polar(1., -std::f64::consts::TAU / len as f64);
    let mut tmp = Complex64::new(1., 0.);
    for _ in 0..len {
        ret.push(tmp);
        tmp *= pri;
    }
    ret
}
