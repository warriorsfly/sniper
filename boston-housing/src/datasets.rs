use csv::Result;
use serde::Deserialize;
#[derive(Debug, Deserialize)]
pub struct House {
    #[serde(rename = "RM")]
    pub rm: f64,
    #[serde(rename = "LSTAT")]
    pub lstat: f64,
    #[serde(rename = "PTRATIO")]
    pub ptratio: f64,
    #[serde(rename = "MEDV")]
    pub medv: f64,
}

impl House {
    pub fn read_from_csv(path: String) -> Result<Vec<House>> {
        let mut houses = vec![];
        let mut rdr = csv::Reader::from_path(path)?;

        for item in rdr.deserialize() {
            let house: House = item?;
            houses.push(house);
        }

        Ok(houses)
    }
}

impl House {
    pub fn into_feature_vector(&self)->Vec<f64>{
        vec![self.rm,self.lstat,self.ptratio]
    }

    pub fn into_targets(&self)->f64{
        self.medv
    }
}
