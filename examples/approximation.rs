extern crate neuroflow;
extern crate time;
extern crate rand;

use neuroflow::FeedForward;

use neuroflow::activators::Type::Tanh;
use rand::{thread_rng, Rng};
use rand::distributions::Uniform;

fn main(){
    let mut nn = FeedForward::new(&[1, 5, 3, 1]);

    let prev = time::now_utc();

    nn.activation(Tanh)
        .learning_rate(0.007)
        .train(|| {
            let mut rnd_range = thread_rng();
            let x = rnd_range.sample(Uniform::new(-3.0, 3.0));
            ([x], [0.5*(x.exp().sin()) - (-x.exp()).cos()])
        }, 60_000);

    let mut res;

    let mut i = 0.0;
    while i <= 0.3{
        res = nn.calc(&[i])[0];
        println!("for [{:.3}], [{:.3}] -> [{:.3}]", i, 0.5*(i.exp().sin()) - (-i.exp()).cos(), res);
        i += 0.07;
    }

    println!("\nSpend time: {:.3}", (time::now_utc() - prev).num_milliseconds() as f64 / 1000.0);
}
