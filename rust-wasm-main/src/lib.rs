mod perceptron;

// use std::{rc::Rc, borrow::Borrow};
use rand::Rng;
use serde_wasm_bindgen::to_value;
use std::rc::Rc;
use wasm_bindgen::JsValue;

use perceptron::perceptron::*;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Wrapper {
    population: Vec<Perceptron>,
    data: Rc<Vec<Vec<f64>>>,
    n_max_iter: usize,
    data_len: usize,
}

#[wasm_bindgen]
impl Wrapper {
    // Vai executar o treinando, retornando o melhor filho daqla geração
    // [[parametros do filho de melhor acuraria, acuracia_do_melhor_filho], ...]
    pub fn train_wrapper(&mut self, n_ger: usize, mutation_rate: f64, crosssover_rate: f64) -> Vec<JsValue> {
        let mut result: Vec<JsValue> = Vec::new();

        for _ in 0..n_ger {
            let mut scores: Vec<f64> = Vec::new();
            let mut best_ind = (0, 0.0);

            // Calcular cada perceptron
            (0..self.population.len())
                .for_each(|i| scores.push(self.population[i].train(self.n_max_iter)));

            scores.into_iter().enumerate().for_each(|(i, accuracy)| {
                if accuracy > best_ind.1 {
                    best_ind = (i, accuracy)
                }
            });

            // result.push(to_value(&best_ind.0).unwrap());
            result.push(to_value(&best_ind.1).unwrap());
            result.push(to_value(&self.population[best_ind.0].get_parameters()).unwrap());
            result.push(to_value(&self.population[best_ind.0].get_weight()).unwrap());
            result.push(to_value(&self.population[best_ind.0].get_bias()).unwrap());

            self.update_wrapper(mutation_rate, crosssover_rate)
        }

        result
    }

    pub fn update_wrapper(&mut self, mutation_rate: f64, crossover_rate: f64) {
        let mut rng = rand::thread_rng();
        let pop_len = self.population.len();

        let total_sum: f64 = self
            .population
            .iter()
            .fold(0.0, |sum, acc| sum + acc.get_accuracy());
        let mut next_gen: Vec<Vec<usize>> = Vec::new();
        let accumulate: Vec<f64> = (0..pop_len).fold(Vec::new(), |mut v, i| {
            match v.last() {
                Some(prev) => v.push(prev + self.population[i].get_accuracy() / total_sum),
                None => v.push(self.population[i].get_accuracy() / total_sum),
            }

            v
        });

        for _ in 0..pop_len {
            let index = accumulate.binary_search_by(|&val| val.partial_cmp(&rng.gen::<f64>()).unwrap())
                .unwrap_or_else(|e| e);

            next_gen.push(self.population[index].get_parameters());
        }
        
        let mut ng: Vec<Vec<usize>> = Vec::new();

        for i in 0..pop_len {
            if crossover_rate < rng.gen::<f64>() {
                let xi = next_gen[rng.gen_range(0..pop_len)].clone();
                let yi = next_gen[rng.gen_range(0..pop_len)].clone();
                
                let mid = rng.gen_range(0..self.data_len);

                let mut left = xi.into_iter().filter(|&x| x < mid).collect::<Vec<usize>>();
                let right = yi.into_iter().filter(|&x| x >= mid).collect::<Vec<usize>>();
                left.extend(right);

                ng.push(left);
            } else {
                ng.push(next_gen[rng.gen_range(0..pop_len)].clone())
            }
        }

        for i in 0..pop_len {
            let mut new_params: Vec<usize> = Vec::new();
            for j in 0..self.data_len {
                if rng.gen::<f64>() < mutation_rate {
                    if let Err(_) = ng[i].binary_search(&j) {
                        new_params.push(j)
                    }
                } else {
                    if let Ok(_) = ng[i].binary_search(&j) {
                        new_params.push(j)
                    }
                }
            }

            self.population[i].update_parameters(new_params);
        }
    }

    // pub fn update_wrapper(&mut self, mutation_rate: f64) {
    // let mut rng = rand::thread_rng();
    // let total_sum: f64 = self
    //     .population
    //     .iter()
    //     .fold(0.0, |sum, acc| sum + acc.get_accuracy());
    // let mut next_gen: Vec<Vec<usize>> = Vec::new();
    // let accumulate: Vec<f64> = (0..self.population.len()).fold(Vec::new(), |mut v, _| {
    //     match v.last() {
    //         Some(prev) => v.push((prev + rng.gen::<f64>()) / total_sum),
    //         None => v.push((rng.gen::<f64>()) / total_sum),
    //     }

    //     v
    // });

    // for _ in 0..accumulate.len() {
    //     let mut index = accumulate
    //         .binary_search_by(|&val| val.partial_cmp(&rng.gen::<f64>()).unwrap()).unwrap();

    //     if index == accumulate.len() {
    //         index -= 1
    //     }

    //     next_gen.push(self.population[index].get_parameters());
    // }

    //     for i in 0..self.population.len() {
    //         let mut new_params: Vec<usize> = Vec::new();

    //         // Mutar
    //         for j in 0..self.data_len {
    //             if rng.gen::<f64>() < mutation_rate {
    //                 if let Err(_) = self.population[j].get_parameters().binary_search(&i) {
    //                     new_params.push(j)
    //                 }
    //             } else {
    //                 if let Ok(_) = self.population[j].get_parameters().binary_search(&i) {
    //                     new_params.push(j)
    //                 }
    //             }
    //         }

    //         self.population[i].update_parameters(new_params);
    //     }
    // }

    pub fn get_wrapper(&self) -> String {
        self.population
            .iter()
            .map(|p| {
                format!(
                    "{:?}",
                    PerceptronForPrint {
                        parameters: p.get_parameters(),
                        weight: p.get_weight(),
                        answ_index: 0,
                        bias: p.get_bias(),
                        step_size: p.get_step_size(),
                        accuracy: p.get_accuracy()
                    }
                )
            })
            .collect::<Vec<String>>()
            .join("\n\n")
    }
}

#[wasm_bindgen]
pub fn new_wrapper_from_string(
    data_str: String,
    pop_len: usize,
    n_params: usize,
    answ_index: usize,
    n_max_iter: usize,
    step_size: f64,
) -> Wrapper {
    let data: Rc<Vec<Vec<f64>>> = ProcessData::evaluate_data(data_str);
    let mut rng = rand::thread_rng();

    Wrapper {
        population: (0..pop_len)
            .map(|_| {
                new_perceptron_to_wrapper(
                    Rc::clone(&data),
                    (0..n_params).fold(Vec::new(), |mut v, i| {
                        if rng.gen_bool(0.7) {
                            v.push(i)
                        }

                        v
                    }),
                    answ_index,
                    step_size,
                )
            })
            .collect(),
        data: Rc::clone(&data),
        n_max_iter: n_max_iter,
        data_len: data[0].len(),
    }

    // Wrapper {
    //     population: (0..pop_len).map(|i| {
    //         new_perceptron_to_wrapper(
    //             Rc::clone(&data),
    //             (0..n_params).collect(),
    //             answ_index,
    //             step_size
    //         )
    //     }).collect(),
    //     data: Rc::clone(&data),
    //     n_max_iter: n_max_iter,
    // }
}
