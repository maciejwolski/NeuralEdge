use crate::backends::Tensor;
use ndarray::{Axis, IxDyn};

use num_traits::{AsPrimitive, Float};

#[derive(Clone)]
pub enum Activation {
    Sigmoid,
    Relu,
    Gelu,
    Softmax
}

pub fn sigmoid<T, B: Tensor<T>>(x: &B) -> B
where
    T: Float,
{
    let result = x.get_data().mapv(|x| x.exp() / (x.exp() + T::one()));
    let out_data_shape = result.shape();
    let out = B::new(result.clone());
    if out.get_data().shape().len() == 1 {
        let data = out.get_data().view().into_shape((1, out_data_shape[0])).unwrap().into_dimensionality::<IxDyn>().unwrap().to_owned();
        let out = B::new(data);
        out
    } else {
        out
    }
}

pub fn sigmoid_prime<T, B: Tensor<T>>(x: &B) -> B
where
    T: Float,
{
    let result = x.get_data().mapv(|x| x * (T::one() - x));
    let out_data_shape = result.shape();
    let out = B::new(result.clone());
    if out.get_data().shape().len() == 1 {
        let data = out.get_data().view().into_shape((1, out_data_shape[0])).unwrap().into_dimensionality::<IxDyn>().unwrap().to_owned();
        let out = B::new(data);
        out
    } else {
        out
    }
}

pub fn relu<T, B: Tensor<T>>(x: &B) -> B
where
    T: Float,
{
    let result = x.get_data().mapv(|x| x.max(T::zero()));
    let out_data_shape = result.shape();
    let out = B::new(result.clone());
    if out.get_data().shape().len() == 1 {
        let data = out.get_data().view().into_shape((1, out_data_shape[0])).unwrap().into_dimensionality::<IxDyn>().unwrap().to_owned();
        let out = B::new(data);
        out
    } else {
        out
    }
}

pub fn relu_prime<T, B: Tensor<T>>(x: &B) -> B
where
    T: Float,
{
    let result = x.get_data().mapv(|x| if x > T::zero() { T::one() } else { T::zero() });
    let out_data_shape = result.shape();
    let out = B::new(result.clone());
    if out.get_data().shape().len() == 1 {
        let data = out.get_data().view().into_shape((1, out_data_shape[0])).unwrap().into_dimensionality::<IxDyn>().unwrap().to_owned();
        let out = B::new(data);
        out
    } else {
        out
    }
}

pub fn softmax<T, B: Tensor<T>>(x: &B) -> B
where
    T: Float,
{
    let data = x.get_data();
    let last_axis = x.get_data().ndim() - 1;

    // Calculate the maximum values along the last axis, ignoring NaN values
    let max_vals = data.map_axis(Axis(last_axis), |row| {
        row.iter()
           .filter_map(|&v| if v.is_nan() { None } else { Some(v) })
           .fold(T::neg_infinity(), |a, b| if a > b { a } else { b })
    });

    // Ensure we handle the case where all values might be NaN
    let max_vals = max_vals.mapv(|v| if v.is_infinite() { T::zero() } else { v });

    // Broadcast max values back to the original shape to subtract from data
    let binding = max_vals.insert_axis(Axis(last_axis));
    let max_vals_broadcasted = binding.broadcast(data.raw_dim()).unwrap();

    let exps = data - &max_vals_broadcasted;
    let exps = exps.mapv(|x| x.exp());

    let sum = exps.sum_axis(ndarray::Axis(last_axis));
    let sum_reshaped = sum.insert_axis(Axis(last_axis));
    let safe_sum_reshaped = sum_reshaped.mapv(|v| if v == T::zero() { T::epsilon() } else {v});
    let result = exps / safe_sum_reshaped;
    let out = B::new(result);

    out
}

pub fn softmax_prime<T, B: Tensor<T>>(x: &B) -> B
where
    T: Float,
{
    let softmax_x = softmax(x);
    let batch_size = softmax_x.shape()[0];
    let dim1 = softmax_x.shape()[1];
    let dim2 = softmax_x.shape()[2];
    let mut softmax_prime_output = B::zeros_like(&softmax_x);

    for n in 0..batch_size {
        for i in 0..dim1 {
            for j in 0..dim2 {
                // Access the softmax probability for the current element
                let s_ij = softmax_x.get_data()[[n, i, j]];

                // Compute the derivative for the diagonal element, s_i * (1 - s_i)
                let derivative_ij = s_ij * (T::one() - s_ij);

                // Store the computed derivative in the corresponding position
                softmax_prime_output.get_data_mut()[[n, i, j]] = derivative_ij;
            }
        }
    }

    softmax_prime_output
}

pub fn gelu<T, B: Tensor<T>>(x: &B) -> B
where
    T: Float + AsPrimitive<f32>,
    B: Clone
{   
    let pi = std::f32::consts::PI;

    let data = x.get_data().mapv(|x| {
        let cdf = 0.5 * (1.0 + ((2.0 / pi).sqrt()) * (x.as_() + 0.044715 * x.as_().powi(3))).tanh();
        x * T::from(cdf).unwrap()
    });
    let mut result = x.clone();
    result.set_data(data);
    result
}

pub fn gelu_prime<T, B: Tensor<T>>(x: &B) -> B
where
    T: Float + AsPrimitive<f32>,
    B: Clone
{
    let pi = std::f32::consts::PI;
    
    let data = x.get_data().mapv(|x| {
        let pdf = (2.0 / pi).sqrt() * (x.as_() + 0.044715 * x.as_().powi(3)).exp() / (1.0 + (x.as_() + 0.044715 * x.as_().powi(3)).exp()).powi(2);
        T::from(0.5 * (1.0 + ((2.0 / pi).sqrt() * (x.as_() + 0.044715 * x.as_().powi(3))).tanh()) + x.as_() * pdf).unwrap()
    });
    let mut result = x.clone();
    result.set_data(data);
    result
}

// Gated Linear Unit
pub fn swiglu<T, B: Tensor<T>>(x: &B, g: &B) -> B
where
    T: Float,
{
    let sigmoid_g = sigmoid(g);  // Compute sigmoid(g(x))
    let result = x.get_data() * sigmoid_g.get_data();  // Element-wise multiplication of x and sigmoid(g(x))
    let out = B::new(result);
    out
}

pub fn swiglu_prime<T, B: Tensor<T>>(x: &B, g: &B, g_prime: &B) -> B
where
    T: Float,
{
    let sigmoid_g = sigmoid(g);  // Compute sigmoid(g(x))
    let sigmoid_prime_g = sigmoid_prime(g);  // Compute the derivative of sigmoid, sigmoid'(g(x))
    let result = sigmoid_g.get_data() + x.get_data() * (sigmoid_prime_g.get_data() * g_prime.get_data());  // Applying the product rule
    let out = B::new(result);
    out
}