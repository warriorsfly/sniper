pub fn r_squared_score(y_test: &Vec<f64>, y_preds: &Vec<f64>) -> f64 {
    let mv: f64 = y_test
        .iter()
        .zip(y_preds.iter())
        .fold(0., |v, (y_i, y_i_hat)| v + (y_i - y_i_hat).powi(2));

    let mean: f64 = y_test.iter().sum();
    let var = y_test.iter().fold(0., |v, &x| v + (x - mean).powi(2));

    let r2: f64 = 1.0 - (mv / var);
    r2
}
