use std::rc::Rc;

use rand::{self, Rng};

// Perceptron
pub struct Perceptron {
    parameters: Vec<usize>,
    weight: Vec<f64>,
    bias: f64,
    data: Rc<Vec<Vec<f64>>>,
    answ_index: usize,
    step_size: f64,
    accuracy: f64,
}

#[derive(Debug)]
pub struct PerceptronForPrint {
    pub parameters: Vec<usize>,
    pub weight: Vec<f64>,
    pub bias: f64,
    pub answ_index: usize,
    pub step_size: f64,
    pub accuracy: f64,
}

pub fn new_perceptron_to_wrapper(
    data: Rc<Vec<Vec<f64>>>,
    params: Vec<usize>,
    answ_index: usize,
    step_size: f64,
) -> Perceptron {
    let params: Vec<usize> = params
        .to_vec()
        .into_iter()
        .filter(|&x| x != answ_index)
        .collect();
    let mut rng = rand::thread_rng();

    Perceptron {
        weight: (0..params.len()).map(|_| rng.gen::<f64>() - 0.5).collect(),
        parameters: params,
        bias: rng.gen::<f64>() - 0.5,
        answ_index,
        data,
        step_size: step_size,
        accuracy: 0.0,
    }
}

// pub fn new_perceptron_from_string(
//     params: &[usize],
//     data_str: String,
//     answ_index: usize,
// ) -> Perceptron {
//     let params: Vec<usize> = params
//         .to_vec()
//         .into_iter()
//         .filter(|&x| x != answ_index)
//         .collect();
//     let data: Rc<Vec<Vec<f64>>> = ProcessData::evaluate_data(data_str);
//     let mut rng = rand::thread_rng();

//     Perceptron {
//         parameters: params.to_vec(),
//         weight: (0..params.len()).map(|_| rng.gen::<f64>() - 0.5).collect(),
//         bias: rng.gen::<f64>() - 0.5,
//         answ_index,
//         data,
//         step_size: 0.01,
//         accuracy: 0.0,
//     }
// }

// pub fn new_perceptron_from_vec(
//     params: &[usize],
//     data: &[f64],
//     answ_index: usize,
//     rows: usize,
//     cols: usize,
// ) -> Perceptron {
//     let params: Vec<usize> = params
//         .to_vec()
//         .into_iter()
//         .filter(|&x| x != answ_index)
//         .collect();
//     let mut rng = rand::thread_rng();

//     Perceptron {
//         parameters: params.to_vec(),
//         weight: (0..params.len()).map(|_| rng.gen::<f64>() - 0.5).collect(),
//         bias: rng.gen::<f64>() - 0.5,
//         answ_index,
//         data: ProcessData::vec_to_matrix(data, rows, cols),
//         step_size: 0.01,
//         accuracy: 0.0,
//     }
// }

impl Perceptron {
    pub fn train(&mut self, n_max_iter: usize) -> f64 {
        let mut last_accuracy = 0.0;
        'l: for _ in 0..n_max_iter {
            last_accuracy = self.eval();
            if last_accuracy == 1.0 {
                break 'l;
            }
        }

        last_accuracy
    }

    fn eval(&mut self) -> f64 {
        let mut sum: f64 = 0.0;

        for i in 0..self.data.len() {
            if self.update(i) {
                sum += 1.0
            }
        }

        self.accuracy = sum / (self.data.len() as f64);
        self.accuracy
    }

    fn update(&mut self, row_index: usize) -> bool {
        let answ: f64 =
            Self::activate_function((0..self.parameters.len()).fold(self.bias, |sum, acc| {
                sum + self.data[row_index][acc] * self.weight[acc]
            }));

        if answ != self.data[row_index][self.answ_index] {
            for i in 0..self.weight.len() {
                self.weight[i] -= self.step_size
                    * self.data[row_index][self.parameters[i]]
                    * (answ - self.data[row_index][self.answ_index])
            }

            self.bias -= self.step_size * (answ - self.data[row_index][self.answ_index]);

            false
        } else {
            true
        }
    }

    pub fn update_parameters(&mut self, new_params: Vec<usize>) {
        let new_params: Vec<usize> = new_params
            .to_vec()
            .into_iter()
            .filter(|&x| x != self.answ_index)
            .collect();
        let mut rng = rand::thread_rng();
        
        self.weight = vec![0.0; new_params.len()];
        self.parameters = new_params;
        self.bias = 0.0;
    }

    fn activate_function(sum: f64) -> f64 {
        if sum > 0.0 {
            1.0
        } else {
            0.0
        }
    }

    pub fn get_parameters(&self) -> Vec<usize> {
        self.parameters.clone()
    }

    pub fn get_weight(&self) -> Vec<f64> {
        self.weight.clone()
    }

    pub fn get_bias(&self) -> f64 {
        self.bias
    }

    pub fn get_step_size(&self) -> f64 {
        self.step_size
    }

    pub fn get_accuracy(&self) -> f64 {
        self.accuracy
    }
}

pub struct ProcessData {}

impl ProcessData {
    fn process(s: &str) -> f64 {
        match s {
            "b" => 1.0,
            "g" => 0.0,
            "F" => 1.0,
            "M" => 2.0,
            "f" => 1.0,
            "t" => 2.0,
            "?" => 0.0,
            num => num.parse().unwrap(),
        }
    }

    pub fn evaluate_data(data: String) -> Rc<Vec<Vec<f64>>> {
        let rows: std::str::Split<&str> = data.split("\n");
        let mut v: Vec<Vec<f64>> = Vec::new();

        Rc::from(
            rows.into_iter()
                .fold(&mut v, |v, r| {
                    let values: Vec<f64> = r.split(",").map(|s| Self::process(s)).collect();
                    v.push(values);
                    v
                })
                .to_owned(),
        )
    }

    // fn vec_to_matrix(data: &[f64], rows: usize, cols: usize) -> Rc<Vec<Vec<f64>>> {
    //     let mut matrix: Vec<Vec<f64>> = vec![vec![0.0; cols]; rows];

    //     for i in 0..rows {
    //         for j in 0..cols {
    //             matrix[i][j] = data[cols * i + j]
    //         }
    //     }

    //     Rc::from(matrix)
    // }
}
