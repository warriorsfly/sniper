use std::vec::Vec;

use crate::{datasets::House, sup_metrics::r_squared_score};
use rand::{seq::SliceRandom, thread_rng};
use rusty_machine::{
    self,
    analysis::score::neg_mean_squared_error,
    learning::lin_reg::LinRegressor,
    linalg::{Matrix, Vector},
    prelude::SupModel,
};

pub fn run() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let fl = "csv/housing.csv";

    let mut data = House::read_from_csv(fl.to_string())?;

    data.shuffle(&mut thread_rng());

    let test_size: f64 = 0.5;
    let test_size: f64 = data.len() as f64 * test_size;

    let test_size = test_size.round() as usize;
    let (test_data, train_data) = data.split_at(test_size);

    let train_size = train_data.len();
    let test_size = test_data.len();

    let x_train: Vec<f64> = train_data
        .iter()
        .flat_map(|r| r.into_feature_vector())
        .collect();

    let y_train: Vec<f64> = train_data.iter().map(|r| r.into_targets()).collect();

    let x_test: Vec<f64> = test_data
        .iter()
        .flat_map(|r| r.into_feature_vector())
        .collect();

    let y_test: Vec<f64> = test_data.iter().map(|r| r.into_targets()).collect();

    let x_train = Matrix::new(train_size, 3, x_train);
    let y_train = Vector::new(y_train);

    let x_test = Matrix::new(test_size, 3, x_test);
    let y_test = Matrix::new(test_size, 1, y_test);

    let mut lin_model = LinRegressor::default();
    lin_model.train(&x_train, &y_train)?;

    let predictions = lin_model.predict(&x_test)?;
    let predictions = Matrix::new(test_size, 1, predictions);

    let acc = neg_mean_squared_error(&predictions, &y_test);
    println!("linear regression error: {:?}", acc);
    println!(
        "linear regression R2 score: {:?}",
        r_squared_score(&y_test.data(), &predictions.data())
    );
    Ok(())
}

#[cfg(test)]
mod test {
    #[test]
    fn test_run() {
        use super::*;
        let _ = run();
    }
}
