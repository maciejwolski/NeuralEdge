use crate::backends::Tensor;
use crate::nn::linear::*;
use rayon::prelude::*;

use std::ops::{AddAssign, SubAssign};
use num_traits::{Float, AsPrimitive};

use std::sync::{Arc, Mutex};
pub trait Node<T, B>
where
    T: Float,
    B: Tensor<T>,
    Self: Send + Sync
{
    fn node_forward(&mut self, input: &B) -> B;
    fn node_backward(&mut self, upstream_grad: &B) -> B;
    fn node_update(&mut self, learning_rate: T);
    fn node_shape(&mut self) -> Vec<usize>;
}

// An example how to implement Node trait for a layer

impl<T, B> Node<T, B> for Linear<T, B>
where
    T: Float + AsPrimitive<f32> + Send + Sync,
    B: Tensor<T> + Clone + AddAssign + SubAssign + Send + Sync
{
    fn node_forward(&mut self, input: &B) -> B {
        self.forward(input)
    }

    fn node_backward(&mut self, upstream_gradient: &B) -> B {
        self.backward(upstream_gradient)
    }

    fn node_update(&mut self, learning_rate: T) {
        self.update(learning_rate)
    }

    fn node_shape(&mut self) -> Vec<usize> {
        let weights = self.weights.lock().unwrap();
        weights.shape()
    }
}

pub struct Graph<T, B>
where
    T: Float,
    B: Tensor<T> + Send + Sync,
{
    pub layers: Vec<Arc<Mutex<dyn Node<T, B> + Send + Sync>>>,
}

impl<T, B> Graph<T, B>
where
    T: Float + AsPrimitive<f32> + Send + Sync,
    B: Tensor<T> + Clone + AddAssign + Send + Sync,
{
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add_layer(&mut self, layer: Arc<Mutex<dyn Node<T, B> + Send + Sync>>) {
        self.layers.push(layer);
    }

    pub fn forward(&mut self, input: &B) -> B {
        self.layers.iter_mut().fold(input.clone(), |acc, layer| {
            let mut layer = layer.lock().unwrap();
            layer.node_forward(&acc)
        })
        // let input_count = input.shape()[0];
        // let inputs = Mutex::new(Vec::with_capacity(input_count));

        // // Split batch inputs to a vector of tensors
        // for b in 0..input_count {
        //     let data = input.slice(0, b, b+1);
        //     inputs.lock().unwrap().push(data);
        // }

        // let layers = Arc::clone(&self.layers);
        // let results: Vec<B> = inputs.into_inner().unwrap().par_iter()
        // .map(|input_clone| {
        //     let mut layers = layers.lock().unwrap();
        //     layers.iter_mut().fold(input_clone.to_owned(), |acc, layer| {
        //         let mut layer = layer.lock().unwrap();
        //         layer.node_forward(&acc)
        //     })
        // })
        // .collect();

        // let concatenated = concatenate(Axis(0), &results.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap();
        // B::new(concatenated)
    }

    pub fn backward(&mut self, targets: &B) {
        let mut grad = targets.clone();
        
        for layer in self.layers.iter_mut().rev() {
            let mut layer = layer.lock().unwrap();
            grad = layer.node_backward(&grad);
        }
        // let output_count = targets.shape()[0];
        // let grads = Mutex::new(Vec::with_capacity(output_count));

        // // Split the targets to a vector of tensors
        // for b in 0..output_count {
        //     let data = targets.slice(0, b, b+1);
        //     grads.lock().unwrap().push(data);
        // }

        // let _results: Vec<B> = grads.into_inner().unwrap().iter()
        //     .map(|grad_clone| {

        //         let mut layers = self.layers.lock().unwrap();
        //         layers.iter_mut().rev().fold(grad_clone.to_owned(), |acc, layer| {
        //             let mut layer = layer.lock().unwrap();
        //             layer.node_backward(&acc)
        //         })
        //     }
        // )
        // .collect();
    }

    pub fn update(&mut self, learning_rate: T) {
        self.layers.par_iter_mut().for_each(|layer| {
            let mut layer = layer.lock().unwrap();
            layer.node_update(learning_rate);
        });
    }
}
