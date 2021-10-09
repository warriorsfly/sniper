use rustymachine_regression::linear_reg::run;

mod datasets;
mod rust_tf;
mod rustymachine_regression;
mod sup_metrics;

fn main() {
    let _ = run();
}
