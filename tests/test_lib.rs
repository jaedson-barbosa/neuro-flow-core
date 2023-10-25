extern crate neuroflow;
extern crate time;
extern crate rand;

use neuroflow::FeedForward;

use rand::{thread_rng, Rng};
use rand::distributions::Uniform;
use neuroflow::estimators;


#[test]
fn xor(){
    const ALLOWED_ERROR: f64 = 0.1; // Max allowed error is 10%
    let mut nn = FeedForward::new(&[2, 4, 1]);
    let sc = &[
        (&[0f64, 0f64], &[0f64]),
        (&[1f64, 0f64], &[1f64]),
        (&[0f64, 1f64], &[1f64]),
        (&[1f64, 1f64], &[0f64]),
    ];
    let prev = time::now_utc();

    let mut k;
    let mut rnd_range = thread_rng();

    nn.learning_rate(0.1).momentum(0.01);
    for _ in 0..30_000{
        k = rnd_range.sample(Uniform::new(0, sc.len()));
        nn.fit(sc[k].0, sc[k].1);
    }

    let mut res;
    for v in sc{
        res = nn.calc(v.0)[0];
        println!("for [{:.3}, {:.3}], [{:.3}] -> [{:.3}]",
                 v.0[0], v.0[1], v.1[0], res);

        if (res - v.1[0]).abs() > ALLOWED_ERROR {
            assert!(false);
        }
    }

    println!("\nSpend time: {:.5}", (time::now_utc() - prev));
    assert!(true);
}

#[test]
fn custom_activation(){
    fn func(_x: f64) -> f64{
        0.0
    }

    fn der_func(_x: f64) -> f64{
        0.0
    }

    let mut nn = FeedForward::new(&[1, 2, 1]);
    nn.custom_activation(func, der_func);

    let before_fit: f64 = nn.calc(&[3.2])[0];

    nn.fit(&[1.0], &[2.1]);

    let after_fit: f64 = nn.calc(&[2.1])[0];
    assert_eq!(before_fit, after_fit);
}

#[test]
fn widrows(){
    let w = estimators::widrows(&[2, 1], 0.1);
    assert_eq!(w, 90f64);
}