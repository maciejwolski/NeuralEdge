use std::collections::HashMap;
use super::Tensor;

use num_traits::{Float, Zero, One, FromPrimitive, AsPrimitive};
use std::ops::{Add, Mul, Neg, Sub, AddAssign, SubAssign, MulAssign};
use std::borrow::BorrowMut;
use std::sync::{Arc, Mutex};
use rand::distributions::uniform::SampleUniform;
use ndarray::ScalarOperand;

pub struct AdamW<T, B: Tensor<T>> 
where 
    T: Float + Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Zero + One + std::fmt::Debug + SampleUniform + ScalarOperand + FromPrimitive + Neg<Output = T>
{
    pub learning_rate: T,
    pub beta1: T,
    pub beta2: T,
    pub epsilon: T,
    pub weight_decay: T,
    pub correct_bias: bool,
    pub state: HashMap<String, (B, B)>,
    pub t: T,
    pub params: HashMap<String, Arc<Mutex<B>>>,
    dtype: std::marker::PhantomData<T>,
}

impl<T: 'static, B: Tensor<T>> AdamW<T, B> 
where 
    T: Float + Clone + AddAssign + MulAssign + SubAssign + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Zero + One + std::fmt::Debug + SampleUniform + ScalarOperand + FromPrimitive + Neg<Output = T> + AsPrimitive<f32>,
    T: Mul<T> + MulAssign<T>,
    B: Clone,
{
    pub fn new(learning_rate: T, beta1: T, beta2: T, epsilon: T, weight_decay: T, correct_bias: bool) -> Self {
        let state = HashMap::new();
        let params = HashMap::new();
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            correct_bias,
            state,
            t: T::from(1.0).unwrap(),
            params,
            dtype: std::marker::PhantomData,
        }
    }

    pub fn optimize(&mut self, tensor_key: &String, grad: &B) {
        let t_zero = T::from(0.0).unwrap();
        let t_one = T::from(1.0).unwrap();

        let binding = self.params.get_mut(tensor_key).unwrap();
        let mut params = binding.lock().unwrap();
    
        if let Some((m_state, v_state)) = self.state.get_mut(tensor_key) {
            for (((param, t), m), v) in params.iter_mut()
                .zip(grad.iter())
                .zip(m_state.borrow_mut().iter_mut())
                .zip(v_state.borrow_mut().iter_mut()) {
    
                // Get the gradient value
                let g = *t;
    
                // Update m and v values based on the gradient
                let m_new = self.beta1 * *m + (t_one - self.beta1) * g;
                let v_new = self.beta2 * *v + (t_one - self.beta2) * g * g;
    
                // Calculate bias-corrected estimates
                let m_hat = m_new / (t_one - self.beta1.powf(self.t));
                let v_hat = v_new / (t_one - self.beta2.powf(self.t));
    
                // Apply weight decay directly to the parameter if necessary
                if self.weight_decay > t_zero {
                    *param *= t_one - self.learning_rate * self.weight_decay;
                }

                // Adjust the parameter based on the AdamW algorithm
                *param -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
    
                // Update the states
                *m = m_new;
                *v = v_new;
            }
        }
    }        

    pub fn optimize_with_logging(&mut self, tensor_key: &String, grad: &B) -> (f32, f32, f32, f32) {
        let t_zero = T::from(0.0).unwrap();
        let t_one = T::from(1.0).unwrap();
        
        let mut avg_grad = 0.0;
        let mut avg_p_update = 0.0;
        let mut avg_m_hat = 0.0;
        let mut avg_v_hat = 0.0;
        let mut count = 0;

        let binding = self.params.get_mut(tensor_key).unwrap();
        let mut params = binding.lock().unwrap();

        if let Some((m_state, v_state)) = self.state.get_mut(tensor_key) {
            for (((param, t), m), v) in params.iter_mut()
                .zip(grad.iter())
                .zip(m_state.borrow_mut().iter_mut())
                .zip(v_state.borrow_mut().iter_mut()) {
    
                // Get the gradient value
                let g = *t;
    
                // Update m and v values based on the gradient
                let m_new = self.beta1 * *m + (t_one - self.beta1) * g;
                let v_new = self.beta2 * *v + (t_one - self.beta2) * g * g;
    
                // Calculate bias-corrected estimates
                let m_hat = m_new / (t_one - self.beta1.powf(self.t));
                let v_hat = v_new / (t_one - self.beta2.powf(self.t));
    
                // Apply weight decay directly to the parameter if necessary
                if self.weight_decay > t_zero {
                    *param *= t_one - self.learning_rate * self.weight_decay;
                }

                // Adjust the parameter based on the AdamW algorithm
                let p_update = self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
                *param -= p_update;
    
                // Update the states
                *m = m_new;
                *v = v_new;

                avg_grad += g.as_();
                avg_p_update += p_update.as_();
                avg_m_hat += m_hat.as_();
                avg_v_hat += v_hat.as_();
                count += 1;
            }
        }

        if count > 0 {
            avg_grad /= count as f32;
            avg_p_update /= count as f32;
            avg_m_hat /= count as f32;
            avg_v_hat /= count as f32;
        }

        (avg_grad, avg_p_update, avg_m_hat, avg_v_hat)
    }
    
    pub fn register_parameters(&mut self, parameter_key: String, tensor: Arc<Mutex<B>>) {
        let tensor_clone = tensor.clone();
        self.params.insert(parameter_key.clone(), tensor);

        let tensor_shape = tensor_clone.lock().unwrap().shape();

        if tensor_shape.len() == 2 {
            self.state.insert(parameter_key.clone(), (B::zeros(&[tensor_shape[0], tensor_shape[1]]), B::zeros(&[tensor_shape[0], tensor_shape[1]])));
        } else if tensor_shape.len() == 3 {
            self.state.insert(parameter_key.clone(), (B::zeros(&[tensor_shape[0], tensor_shape[1], tensor_shape[2]]), B::zeros(&[tensor_shape[0], tensor_shape[1], tensor_shape[2]])));
        }
    }

    pub fn zero_grad(&mut self) {
        for (_, (m, v)) in self.state.iter_mut() {
            *m = B::zeros_like(m);
            *v = B::zeros_like(v);
        }
    }

    pub fn step(&mut self, grads: &mut HashMap<String, B>) {
        for (key, grad) in grads {
            self.optimize(key, &grad);
        }
        self.t += T::from(1.0).unwrap();
    }

    pub fn step_with_logging(&mut self, grads: &mut HashMap<String, B>) {
        let mut avg_grad = 0.0;
        let mut avg_p_update = 0.0;
        let mut avg_m_hat = 0.0;
        let mut avg_v_hat = 0.0;
        let mut count = 0;
    
        for (key, grad) in grads {
            let (grad, p_update, m_hat, v_hat) = self.optimize_with_logging(key, &grad);
    
            avg_grad += grad;
            avg_p_update += p_update;
            avg_m_hat += m_hat;
            avg_v_hat += v_hat;
            count += 1;
        }

        self.t += T::from(1.0).unwrap();
    
        if count > 0 {
            avg_grad /= count as f32;
            avg_p_update /= count as f32;
            avg_m_hat /= count as f32;
            avg_v_hat /= count as f32;

            println!("Current Step: {:?}", self.t);
            println!("Average Gradient: {}", avg_grad);
            println!("Average Parameter Update: {}", avg_p_update);
            println!("Average First Moment Estimate: {}", avg_m_hat);
            println!("Average Second Moment Estimate: {}", avg_v_hat);
            println!("==============================");
        }
    }
}

impl<T: 'static, B: Tensor<T>> Default for AdamW<T, B> 
where 
    T: Float + Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Zero + One + std::fmt::Debug + SampleUniform + ScalarOperand + FromPrimitive + Neg<Output = T>
{
    fn default() -> Self {
        Self {
            learning_rate: T::from(0.1).unwrap(),
            beta1: T::from(0.9).unwrap(),
            beta2: T::from(0.98).unwrap(),
            epsilon: T::from(1e-9).unwrap(),
            weight_decay: T::from(0.01).unwrap(),
            correct_bias: true,
            state: HashMap::new(),
            t: T::from(1.0).unwrap(),
            params: HashMap::new(),
            dtype: std::marker::PhantomData,
        }
    }
}

pub struct RMSProp<T, B: Tensor<T>> 
where 
    T: Float + Clone + AddAssign + MulAssign + FromPrimitive + Zero + One + ScalarOperand + std::fmt::Debug,
{
    pub learning_rate: T,
    pub decay_rate: T,
    pub epsilon: T,
    pub state: HashMap<String, B>,
    pub params: HashMap<String, Arc<Mutex<B>>>,
    dtype: std::marker::PhantomData<T>,
}

impl<T: 'static, B: Tensor<T>> RMSProp<T, B> 
where 
    T: Float + Clone + AddAssign + MulAssign + SubAssign + FromPrimitive + Zero + One + ScalarOperand + std::fmt::Debug,
    B: Clone,
{
    pub fn new(learning_rate: T, decay_rate: T, epsilon: T) -> Self {
        Self {
            learning_rate,
            decay_rate,
            epsilon,
            state: HashMap::new(),
            params: HashMap::new(),
            dtype: std::marker::PhantomData,
        }
    }

    pub fn optimize(&mut self, tensor_key: &String, grad: &B) {
        let t_one = T::one();

        let binding = self.params.get_mut(tensor_key).unwrap();
        let mut params = binding.lock().unwrap();

        let state_entry = self.state.entry(tensor_key.clone()).or_insert_with(|| B::zeros_like(grad));

        for ((param, grad_value), state_value) in params.iter_mut().zip(grad.iter()).zip(state_entry.borrow_mut().iter_mut()) {
            // Update the moving average of the squared gradients
            *state_value = self.decay_rate * *state_value + (t_one - self.decay_rate) * *grad_value * *grad_value;

            // Update parameters
            *param -= self.learning_rate * *grad_value / (state_value.sqrt() + self.epsilon);
        }
    }

    pub fn step(&mut self, grads: &mut HashMap<String, B>) {
        for (key, grad) in grads {
            self.optimize(key, &grad);
        }
    }

    pub fn register_parameters(&mut self, parameter_key: String, tensor: Arc<Mutex<B>>) {
        let tensor_clone = tensor.clone();
        self.params.insert(parameter_key.clone(), tensor);

        let tensor_shape = tensor_clone.lock().unwrap().shape();

        // Ensure the state map contains a zero-initialized tensor for each parameter tensor
        self.state.entry(parameter_key).or_insert_with(|| B::zeros(&tensor_shape));
    }

    pub fn zero_grad(&mut self) {
        for state in self.state.values_mut() {
            *state = B::zeros_like(state);
        }
    }
}

impl<T: 'static, B: Tensor<T>> Default for RMSProp<T, B> 
where 
    T: Float + Clone + AddAssign + MulAssign + FromPrimitive + Zero + One + ScalarOperand + std::fmt::Debug,
{
    fn default() -> Self {
        Self {
            learning_rate: T::from(0.001).unwrap(),
            decay_rate: T::from(0.9).unwrap(),
            epsilon: T::from(1e-8).unwrap(),
            state: HashMap::new(),
            params: HashMap::new(),
            dtype: std::marker::PhantomData,
        }
    }
}