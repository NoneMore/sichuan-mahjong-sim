//! One-off implementation-milestone measurement; see docs/validation.md.
//! This is not a production benchmark framework or a Cargo example.

use sichuan_mahjong_sim::policy::{
    random_seat_seed, FirstLegalPolicy, HeuristicPolicy, Policy, RandomPolicy,
};
use sichuan_mahjong_sim::runner::Runner;
use sichuan_mahjong_sim::{Action, Decision, Seat};
use std::cell::Cell;
use std::rc::Rc;
use std::time::{Duration, Instant};

// Every seat has its own timing-only counter. No policy can read another seat's
// counter, observations or concealed information; these counters never affect choices.
struct TimedPolicy {
    inner: Box<dyn Policy>,
    timing: Rc<Cell<(u64, Duration)>>,
}
impl Policy for TimedPolicy {
    fn choose(&mut self, decision: &Decision<'_>) -> Action {
        let start = Instant::now();
        let action = self.inner.choose(decision);
        let elapsed = start.elapsed();
        let (n, total) = self.timing.get();
        self.timing.set((n + 1, total + elapsed));
        action
    }
}

fn main() {
    let other = std::env::args().nth(1).expect("first or random");
    assert!(other == "first" || other == "random");
    let pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
    let start = Instant::now();
    let mut policy_time = Duration::ZERO;
    let mut heuristic_time = Duration::ZERO;
    let mut heuristic_decisions = 0u64;
    let mut rules_time = Duration::ZERO;
    let mut decisions = 0u64;
    let mut scores = [0i64; 4];
    let mut games = 0;
    for seed in 1u64..=4 {
        for pair in pairs {
            let timings: [Rc<Cell<(u64, Duration)>>; 4] =
                std::array::from_fn(|_| Rc::new(Cell::new((0, Duration::ZERO))));
            let policies: [Box<dyn Policy>; 4] = std::array::from_fn(|i| {
                let inner: Box<dyn Policy> = if pair.contains(&i) {
                    Box::new(HeuristicPolicy::new())
                } else if other == "first" {
                    Box::new(FirstLegalPolicy::new())
                } else {
                    Box::new(RandomPolicy::new(random_seat_seed(
                        seed,
                        Seat::new(i).unwrap(),
                    )))
                };
                Box::new(TimedPolicy {
                    inner,
                    timing: timings[i].clone(),
                }) as Box<dyn Policy>
            });
            let report = Runner::new(policies)
                .run_new(seed)
                .expect("legal completed game");
            assert_eq!(report.scores.iter().sum::<i32>(), 0);
            for i in 0..4 {
                scores[i] += i64::from(report.scores[i]);
                if pair.contains(&i) {
                    let (n, elapsed) = timings[i].get();
                    heuristic_decisions += n;
                    heuristic_time += elapsed;
                }
            }
            decisions += report.decisions;
            policy_time += report.policy_time;
            rules_time += report.rules_time;
            games += 1;
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!("heuristic/{other}: seeds=1,2,3,4; policy_seed=0; six pairs; fresh policies per game");
    println!("games={games} decisions={decisions} scores={scores:?} total_s={elapsed:.6} policy_s={:.6} rules_s={:.6} games_per_s={:.3} decisions_per_s={:.3} mixed_policy_us_per_decision={:.3}", policy_time.as_secs_f64(), rules_time.as_secs_f64(), games as f64 / elapsed, decisions as f64 / elapsed, policy_time.as_secs_f64() * 1e6 / decisions as f64);
    println!("heuristic_decisions={heuristic_decisions} heuristic_s={:.6} heuristic_us_per_decision={:.3}", heuristic_time.as_secs_f64(), heuristic_time.as_secs_f64() * 1e6 / heuristic_decisions as f64);
}
