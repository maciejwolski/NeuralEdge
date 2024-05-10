use crate::backends::Tensor;
use super::activations::*;

use num_traits::{AsPrimitive, Float};
use std::ops::{AddAssign, SubAssign};
use std::sync::{Arc, Mutex};

use ndarray::{s, Ix1, Array1};

pub struct Linear<T, B>
where
    T: Float,
    B: Tensor<T>
{
    pub weights: Arc<Mutex<B>>,
    pub bias: Arc<Mutex<B>>,
    activation: fn(&B) -> B,
    activation_prime: fn(&B) -> B,
    input: B,
    output: B,
    pub weights_grad: B,
    pub bias_grad: B,
    dtype: std::marker::PhantomData<T>,
}

impl<T, B> Linear<T,B> 
where
    T: Float + AsPrimitive<f32>,
    B: Tensor<T> + Clone + AddAssign + SubAssign,
{
    pub fn new(in_features: usize, out_features: usize, activations: Activation) -> Self {

        let weights: Arc<Mutex<B>>;

        match activations {
            Activation::Sigmoid => {
                weights = Arc::new(Mutex::new(B::glorot_uniform(&[in_features, out_features])));
            }
            Activation::Relu => {
                weights = Arc::new(Mutex::new(B::he_uniform(&[in_features, out_features])));
            }
            Activation::Gelu => {
                weights = Arc::new(Mutex::new(B::glorot_uniform(&[in_features, out_features])));
            }
            Activation::Softmax => {
                weights = Arc::new(Mutex::new(B::glorot_uniform(&[in_features, out_features])));
            }
        }

        let bias = Arc::new(Mutex::new(B::zeros(&[1, out_features])));

        let activation_func: fn(&B) -> B;
        let activation_prime: fn(&B) -> B;

        match activations {
            Activation::Sigmoid => {
                activation_func = sigmoid;
                activation_prime = sigmoid_prime;
            }
            Activation::Relu => {
                activation_func = relu;
                activation_prime = relu_prime;
            }
            Activation::Gelu => {
                activation_func = gelu;
                activation_prime = gelu_prime;
            }
            Activation::Softmax => {
                activation_func = softmax;
                activation_prime = softmax_prime;
            }
        }

        Self {
            weights,
            bias,
            activation: activation_func,
            activation_prime,
            input: B::zeros(&[in_features]),
            output: B::zeros(&[out_features]),
            weights_grad: B::zeros(&[in_features, out_features]),
            bias_grad: B::zeros(&[out_features]),
            dtype: std::marker::PhantomData,
        }
    }

    pub fn from_checkpoint(weights: B, bias: B, activations: Activation) -> Self {
        let activation_func: fn(&B) -> B;
        let activation_prime: fn(&B) -> B;

        match activations {
            Activation::Sigmoid => {
                activation_func = sigmoid;
                activation_prime = sigmoid_prime;
            }
            Activation::Relu => {
                activation_func = relu;
                activation_prime = relu_prime;
            }
            Activation::Gelu => {
                activation_func = gelu;
                activation_prime = gelu_prime;
            }
            Activation::Softmax => {
                activation_func = softmax;
                activation_prime = softmax_prime;
            }
        }

        let input_dim = weights.shape()[0];
        let output_dim = weights.shape()[1];

        let weights_grad = B::zeros_like(&weights);
        let bias_grad = B::zeros_like(&bias);

        Self {
            weights: Arc::new(Mutex::new(weights)),
            bias: Arc::new(Mutex::new(bias)),
            activation: activation_func,
            activation_prime,
            input: B::zeros(&[input_dim]),
            output: B::zeros(&[output_dim]),
            weights_grad,
            bias_grad,
            dtype: std::marker::PhantomData,
        }
    }

    pub fn forward(&mut self, input: &B) -> B {
        let weights = self.weights.lock().unwrap();
        let bias = self.bias.lock().unwrap();
        let mut result = input.matmul(&weights);

        self.input = input.clone();

        if result.get_data().ndim() == 2 {
            result += bias.clone();
        } else {
            let bias_dim = Array1::from(bias.index_axis_move(0, 0).get_data().clone().into_dimensionality::<Ix1>().unwrap());
            for _ in 0..result.shape()[0] {
                result.add_axis(0, &bias_dim);
            }
        }

        let out = (self.activation)(&mut result);
        self.output = out.clone();
        out
    }

    pub fn backward(&mut self, upstream_gradient: &B) -> B {
        let activations = (self.activation_prime)(&mut self.output);
        let mut delta = upstream_gradient.mul(&activations);

        let weights = self.weights.lock().unwrap();

        // Gradient clipping
        let clip_value = T::from(1.0).unwrap();
        let norm = delta.norm();
        if norm > clip_value {
            delta = delta.mul_scalar(clip_value / norm);
        }

        let self_input_dims = self.input.shape().len();
        let delta_dims = delta.shape().len();

        let weights_grad: B;
        let bias_grad: B;

        if delta_dims == 3 {
            let batch_size = T::from(upstream_gradient.shape()[0]).unwrap();
            let delta_summed = delta.sum_axis(0);

            if self_input_dims == 3 {
                weights_grad = self.input.transpose_axes(1, 2).matmul(&delta_summed).div_scalar(batch_size);
            } else {
                weights_grad = self.input.transpose().matmul(&delta_summed).div_scalar(batch_size);
            }
            bias_grad = delta_summed.sum_axis(0).div_scalar(batch_size);
        } else {
            weights_grad = self.input.transpose().matmul(&delta);
            bias_grad = delta.sum_axis(0);
        }

        let input_grad = delta.matmul(&weights.transpose());

        let avg_weights_grad = weights_grad.mean_axis(0);

        self.weights_grad = avg_weights_grad;
        self.bias_grad = bias_grad;

        input_grad
    }

    pub fn update(&mut self, learning_rate: T) {
        if !self.weights_grad.iter().any(|&x| x.is_nan()) {
            let mut weights = self.weights.lock().unwrap();
            let mut bias = self.bias.lock().unwrap();
            let grad_weights = self.weights_grad.mul_scalar(learning_rate);
            (*weights) -= grad_weights;
            let grad_bias = self.bias_grad.mul_scalar(learning_rate);
            (*bias) -= grad_bias;
        }
    }
}

pub struct Sequential<T, B>
where
    T: Float + AsPrimitive<f32>,
    B: Tensor<T> + Clone
{
    pub layers: Vec<Arc<Mutex<Linear<T,B>>>>,
}

impl<T, B> Sequential<T,B>
where
    T: Float + AsPrimitive<f32>,
    B: Tensor<T> + AddAssign + SubAssign + Clone,
{
    pub fn new() -> Self {
        Self {
            layers: Vec::new()
        }
    }

    pub fn add_layer(&mut self, layer: Box<Linear<T, B>>) {
        let layer_mutex = Arc::new(Mutex::new(*layer));
        self.layers.push(layer_mutex);
    }

    pub fn forward(&mut self, input: &B) -> B {
        let mut result = B::zeros_like(input);

        for i in 0..input.shape()[0] {
            let mut temp_input = input.index_axis_move(0, i);

            for layer_mutex in &mut self.layers {
                let mut layer = layer_mutex.lock().unwrap();
                temp_input = layer.forward(&temp_input);
            }
            result.get_data_mut().slice_mut(s![i, .., ..]).assign(&temp_input.get_data());
        }

        result
    }

    pub fn backward(&mut self, output_grad: &B) -> B {
        let mut current_grad = output_grad.clone();
        for layer_mutex in self.layers.iter_mut().rev() {
            let mut layer = layer_mutex.lock().unwrap();
            current_grad = layer.backward(&current_grad);
        }

        current_grad
    }

    pub fn update(&mut self, learning_rate: T) {
        for layer_mutex in self.layers.iter_mut() {
            let mut layer = layer_mutex.lock().unwrap();
            layer.update(learning_rate);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use crate::backends::cpu::CpuTensor;

    #[test]
    fn test_linear_forward() {
        let mut linear = Linear::new(3, 2, Activation::Sigmoid);
        let input = CpuTensor::new(array![[0.1, 0.2, 0.3]].into_dyn());
        linear.forward(&input);
    }

    #[test]
    fn test_linear_backward() {
        let mut linear = Linear::new(6, 5, Activation::Sigmoid);
        let input = CpuTensor::new(array![[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]].into_dyn());
        linear.forward(&input);
        let upstream_gradient = CpuTensor::new(array![[0.1, 0.1, 0.1, 0.1, 0.1]].into_dyn());
        linear.backward(&upstream_gradient);
    }

    #[test]
    fn test_sequential_forward() {
        let mut model = Sequential::<f32, CpuTensor<f32>>::new();
        model.add_layer(Box::new(Linear::new(3, 2, Activation::Sigmoid)));
        model.add_layer(Box::new(Linear::new(2, 1, Activation::Sigmoid)));
        let input = Tensor::<f32>::random(&[1, 3, 3]);
        model.forward(&input);
    }

    #[test]
    fn test_sequential_backward() {
        let mut model = Sequential::<f32, CpuTensor<f32>>::new();
        model.add_layer(Box::new(Linear::new(32, 64, Activation::Sigmoid)));
        model.add_layer(Box::new(Linear::new(64, 32, Activation::Sigmoid)));
        let input = Tensor::random(&[1, 3, 32]);
        model.forward(&input);
        let upstream_gradient = Tensor::<f32>::new(ndarray::Array::from_elem((1, 32), 0.1).into_dyn());
        model.backward(&upstream_gradient);
    }
}