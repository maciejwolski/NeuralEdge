use crate::backends::Tensor;
use crate::nn::linear::*;
use rayon::prelude::*;

use std::{fmt::Debug, ops::{AddAssign, SubAssign}};
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
    T: Float + AsPrimitive<f32> + Send + Sync + Debug,
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
    T: Float + AsPrimitive<f32> + Send + Sync + Debug,
    B: Tensor<T> + Clone + AddAssign + Send + Sync,
{
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add_layer(&mut self, layer: Arc<Mutex<dyn Node<T, B> + Send + Sync>>) {
        self.layers.push(layer);
    }

    pub fn forward(&mut self, input: &B) -> B {
        //let mut reversed_labels = vec!["logits", "layer_norm2", "feedforward2", "feedforward1", "layer_norm1", "multiattention", "tok_embed", "input"];
        self.layers.iter_mut().fold(input.clone(), |acc, layer| {
            //println!("{}: {:?}", reversed_labels.pop().unwrap(), acc.get_data());
            //std::thread::sleep(std::time::Duration::from_millis(1500));
            let mut layer = layer.lock().unwrap();
            layer.node_forward(&acc)
        })
    }

    pub fn backward(&mut self, targets: &B) {
        let mut grad = targets.clone();

        //let mut layer_labels = vec!["tok_embed", "multiattention", "layer_norm1", "feedforward1", "feedforward2", "layer_norm2", "logits", "output"];
        for layer in self.layers.iter_mut().rev() {
            //println!("{}: {:?}", layer_labels.pop().unwrap(), grad.get_data());
            //std::thread::sleep(std::time::Duration::from_millis(1500));
            let mut layer = layer.lock().unwrap();
            grad = layer.node_backward(&grad);
        }
    }

    pub fn update(&mut self, learning_rate: T) {
        self.layers.par_iter_mut().for_each(|layer| {
            let mut layer = layer.lock().unwrap();
            layer.node_update(learning_rate);
        });
    }
}
