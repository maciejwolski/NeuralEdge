use super::Tensor;
use crate::neuraledge_core::nn::linear::{Linear, Sequential};
use crate::neuraledge_core::nn::activations::{Activation, softmax, softmax_prime};
use crate::neuraledge_core::backends::graph::Node;

use super::{Transformer, validate_gradients};

use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, MulAssign, SubAssign};
use std::sync::{Arc, Mutex};
use num_traits::{FromPrimitive, AsPrimitive, Float};
use ndarray::{s, Array2, Axis, ScalarOperand};
use serde::{Serialize, Deserialize};

use rand::distributions::uniform::SampleUniform;
use rand::distributions::{Uniform, Distribution};
use rand::thread_rng;

// --------------- EMBEDDINGS & POSITIONAL ENCODING ---------------

pub struct Embedding<T,B>
where
    T: Float,
    B: Tensor<T>
{
    pub embedding_dim: usize,
    pub weights: Arc<Mutex<B>>,
    pub inputs: Option<B>,
    pub grads: Option<B>,
    pub pos_encoder: RotaryPositionalEncoding<T,B>,
    dtype: std::marker::PhantomData<T>,
}

impl<T,B> Embedding<T,B>
where
    T: Float + SampleUniform + AsPrimitive<usize> + AddAssign,
    B: Tensor<T> + Clone + SubAssign
{
    pub fn new(vocab_size: usize, embedding_dim: usize, pos_encoder: RotaryPositionalEncoding<T,B>) -> Self {

        let limit = T::from(1.0 / (embedding_dim as f32).sqrt()).unwrap();
        let uniform = Uniform::new(-limit, limit);
        let rng = thread_rng();
        let data_vec: Vec<T> = uniform.sample_iter(rng).take(vocab_size*embedding_dim).collect();
        let data = Array2::from_shape_vec((vocab_size, embedding_dim), data_vec).unwrap().into_dyn();

        Self {
            embedding_dim,
            weights: Arc::new(Mutex::new(B::new(data))),
            inputs: None,
            grads: None,
            pos_encoder,
            dtype: std::marker::PhantomData,
        }
    }

    pub fn from_checkpoint(weights: B, embedding_dim: usize, pos_encoder: RotaryPositionalEncoding<T,B>) -> Self {
        Self {
            embedding_dim,
            weights: Arc::new(Mutex::new(weights)),
            inputs: None,
            grads: None,
            pos_encoder,
            dtype: std::marker::PhantomData,
        }
    }

    pub fn init_inputs(&mut self, batch_size: usize, seq_len: usize) {
        self.inputs = Some(B::zeros(&[batch_size, seq_len]));
    }

    pub fn lookup(&self, token_id: usize) -> B {
        let embedding_weights = self.weights.lock().unwrap();
        let embed = embedding_weights.index_axis(0, token_id);

        B::new(embed.to_owned())
    }

    pub fn get_embeddings(&mut self, tokens: B, batch_id: usize) -> B {
        self.inputs.as_mut().unwrap().get_data_mut().slice_mut(s![batch_id, ..]).assign(&tokens.index_axis_move(0, 0).get_data());

        let mut embeddings = Vec::new();
        for token in tokens.get_data().iter() {
            embeddings.push(self.lookup(token.as_()).clone());
        }

        B::stack(embeddings.iter().collect::<Vec<_>>().as_slice(), 0)
    }

    pub fn forward(&mut self, input: &B) -> B {
        let mut encoded_batch = B::zeros(&[input.shape()[0], input.shape()[1], self.embedding_dim]);
        self.init_inputs(input.shape()[0], input.shape()[1]);

        for i in 0..input.shape()[0] {
            let single_input = input.slice(0, i, i+1);
            let mut embeddings = self.get_embeddings(single_input, i);
            embeddings.insert_axis(0);
            self.pos_encoder.encode(input.shape()[1]);
            self.pos_encoder.tensor.insert_axis(0);
            let pos_tensor = self.pos_encoder.tensor.clone();
            let encoded_input = embeddings.add(&pos_tensor);
            encoded_batch.get_data_mut().slice_mut(s![i, .., ..]).assign(&encoded_input.index_axis_move(0, 0).get_data());
        }

        encoded_batch
    }

    pub fn backward(&mut self, upstream_grads: &B) -> B {
        let weights = self.weights.lock().unwrap();

        let mut binding = B::zeros_like(&weights);
        let embed_grads = binding.get_data_mut();
    
        let input_ids = self.inputs.as_ref().unwrap();

        for i in 0..upstream_grads.shape()[0] {
            let single_grad = upstream_grads.index_axis_move(0, i); // batch >> (seq_len, d_model)
            let single_data = single_grad.get_data();
            for j in 0..upstream_grads.shape()[1] {
                let token_id: usize = input_ids.get_data()[[i, j]].as_();

                for k in 0..upstream_grads.shape()[2] {
                    embed_grads[[token_id, k]] += single_data[[j, k]];
                }
            }
        }

        self.grads = Some(B::new(embed_grads.clone()));

        // return just to satisfy the Node trait
        B::zeros(&[0])
    }

    pub fn update(&mut self, learning_rate: T) {
        let grads = self.grads.as_ref().unwrap();
        let avg_grads = grads.mean_axis(0);

        let mut weights = self.weights.lock().unwrap();
        (*weights) -= avg_grads.mul_scalar(learning_rate);
    }
}

impl<T, B> Node<T, B> for Embedding<T, B>
where
    T: Float + AddAssign + SampleUniform + AsPrimitive<usize> + Send + Sync,
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

pub struct PositionalEncoding<T,B>
where
    T: Float,
    B: Tensor<T>,
{
    embedding_dim: usize,
    pub tensor: B,
    dtype: std::marker::PhantomData<T>
}

#[allow(dead_code)]
impl<T,B> PositionalEncoding<T,B> 
where
    T: Float,
    B: Tensor<T> + Clone,
{
    pub fn new(embedding_dim: usize) -> Self {
        Self { embedding_dim, tensor: Tensor::zeros(&[1, embedding_dim]), dtype: std::marker::PhantomData }
    }

    fn calculate(&self, position: usize, i: usize) -> f32 {
        if i % 2 == 0 {
            (position as f32 / 10000.0f32.powf(i as f32 / 2.0)).sin()
        } else {
            (position as f32 / 10000.0f32.powf(i as f32 / 2.0)).cos()
        }
    }

    pub fn encode(&mut self, seq_len: usize) {
        let mut encoding = Array2::<T>::zeros((seq_len, self.embedding_dim));
        for pos in 0..seq_len {
            for i in 0..self.embedding_dim {
                encoding[(pos, i)] = T::from(self.calculate(pos, i)).unwrap();
            }
        }
        self.tensor = B::new(encoding.into_dyn());
    }
}

#[derive(Clone)]
pub struct RotaryPositionalEncoding<T, B>
where
    T: Float,
    B: Tensor<T>,
{
    embedding_dim: usize,
    pub tensor: B,
    dtype: std::marker::PhantomData<T>
}

impl<T, B> RotaryPositionalEncoding<T, B> 
where
    T: Float,
    B: Tensor<T> + Clone,
{
    pub fn new(embedding_dim: usize) -> Self {
        assert!(embedding_dim % 2 == 0, "Embedding dimension must be even for RoPE.");
        Self { 
            embedding_dim, 
            tensor: Tensor::zeros(&[1, embedding_dim]),
            dtype: std::marker::PhantomData 
        }
    }

    fn get_angles(&self, position: usize, i: usize) -> T {
        // Assumes embedding dimension is even and each pair of dimensions gets its own angle
        T::from(position as f32 / 10000.0f32.powf((i / 2) as f32 / (self.embedding_dim as f32 / 2.0))).unwrap()
    }

    pub fn encode(&mut self, seq_len: usize) {
        let mut sin_cos = Array2::<T>::zeros((seq_len, self.embedding_dim));
        for pos in 0..seq_len {
            for i in (0..self.embedding_dim).step_by(2) { // step by 2 for pairs
                let angle = self.get_angles(pos, i);
                sin_cos[(pos, i)] = angle.cos();
                sin_cos[(pos, i + 1)] = angle.sin();
            }
        }
        self.tensor = B::new(sin_cos.into_dyn());
    }
}

// --------------- LAYERNORM & ATTENTION ---------------

pub struct LayerNorm<T,B>
where
    T: Float,
    B: Tensor<T>,
{
    pub weights: Arc<Mutex<B>>,
    pub bias: Arc<Mutex<B>>,
    input: B,
    output: B,
    residual: B,
    pub weights_grad: B,
    pub bias_grad: B,
    dtype: std::marker::PhantomData<T>
}

impl<T,B> LayerNorm<T,B>
where
    T: Float + FromPrimitive,
    B: Tensor<T> + Clone + SubAssign,
{
    pub fn new(input_dim: usize) -> Self {
        let weights = Arc::new(Mutex::new(B::glorot_uniform(&[1, input_dim])));
        let bias = Arc::new(Mutex::new(B::zeros(&[1, input_dim])));
        let input = B::zeros(&[1, input_dim]);
        let output = B::zeros(&[1, input_dim]);
        let residual = B::zeros(&[1, input_dim]);
        let weights_grad = B::zeros(&[1, input_dim]);
        let bias_grad = B::zeros(&[1, input_dim]);

        Self {
            weights,
            bias,
            input,
            output,
            residual,
            weights_grad,
            bias_grad,
            dtype: std::marker::PhantomData,
        }
    }

    pub fn forward(&mut self, input: &B) -> B {
        self.input = input.clone();
        let epsilon = T::from(1e-5).unwrap();

        // Residual connection: Add the original input to the normalized output
        let input = input.add(&self.residual);

        // Compute mean and variance along the specified axis
        let mean = B::new(input.get_data().mean_axis(Axis(2)).unwrap());
        let variance = input.variance(2);

        // Broadcast the mean and variance for element-wise operations
        let mean_reshaped = mean.get_data().clone().insert_axis(Axis(2));
        let mean_broadcasted = B::new(mean_reshaped.broadcast(input.get_data().raw_dim()).unwrap().to_owned());

        let variance_reshaped = variance.get_data().clone().insert_axis(Axis(2));
        let variance_broadcasted = B::new(variance_reshaped.broadcast(input.get_data().raw_dim()).unwrap().to_owned());

        // Normalize the input using the mean and variance, adjusting variance to avoid zero division
        let adjusted_variance = variance_broadcasted.add_scalar(epsilon);
        let std_dev = adjusted_variance.sqrt();

        let normalized = (input.sub(&mean_broadcasted)).div(&std_dev);

        // Scale and shift the normalized data
        let weights = self.weights.lock().unwrap();
        let bias = self.bias.lock().unwrap();
        let output = normalized.mul(&weights).add(&bias);
        self.output = output.clone();

        output
    }

    pub fn backward(&mut self, upstream_grad: &B) -> B {
        let input_mean = self.input.mean_axis(0);
        let input_var = self.input.variance(0);
        let input_std = input_var.add_scalar(T::from(1e-5).unwrap()).sqrt();
        let normalized_input = (self.input.sub(&input_mean)).div(&input_std);

        let weights = self.weights.lock().unwrap();

        let input_grad_normalized = upstream_grad.mul(&weights).div(&input_std);
        let scale_grad = upstream_grad.mul(&normalized_input).sum_axis(0);
        let bias_grad = upstream_grad.sum_axis(0);

        let mut input_grad = input_grad_normalized.clone();
        input_grad = input_grad.add(&self.residual);

        self.weights_grad = scale_grad;
        self.bias_grad = bias_grad;

        input_grad
    }

    pub fn update(&mut self, learning_rate: T) {

        let mut weights = self.weights.lock().unwrap();
        let mut bias = self.bias.lock().unwrap();

        let avg_scale_grad = self.weights_grad.mean_axis(0);
        let avg_bias_grad = self.bias_grad.mean_axis(0);

        (*weights) -= avg_scale_grad.mul_scalar(learning_rate);
        (*bias) -= avg_bias_grad.mul_scalar(learning_rate);
    }
}

impl<T, B> Node<T, B> for LayerNorm<T, B>
where
    T: Float + FromPrimitive + AsPrimitive<f32> + Send + Sync,
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

pub struct AttentionHead<T,B>
where
    T: Float,
    B: Tensor<T>,
{
    input: Option<B>,
    pub q: Arc<Mutex<B>>,
    pub k: Arc<Mutex<B>>,
    pub v: Arc<Mutex<B>>,
    q_proj: Option<B>,
    k_proj: Option<B>,
    v_proj: Option<B>,
    o_proj: Option<B>,
    pub q_grad: Option<B>,
    pub k_grad: Option<B>,
    pub v_grad: Option<B>,
    //pub o_grad: Option<B>,
    pub output_weights: B,
    output: Option<B>,
    dtype: std::marker::PhantomData<T>
}

impl<T,B> AttentionHead<T,B> 
where
    T: Float + Debug,
    B: Tensor<T> + Clone + AddAssign + Add<Output = B> + SubAssign,
{
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        let q = Arc::new(Mutex::new(B::glorot_uniform(&[d_model / num_heads, d_model / num_heads])));
        let k = Arc::new(Mutex::new(B::glorot_uniform(&[d_model / num_heads, d_model / num_heads])));
        let v = Arc::new(Mutex::new(B::glorot_uniform(&[d_model / num_heads, d_model / num_heads])));
        let output_weights = B::glorot_uniform(&[d_model / num_heads, d_model / num_heads]);
        Self { input: None, q, k, v, q_grad: None, k_grad: None, v_grad: None, q_proj: None, k_proj: None, v_proj: None, o_proj: None, output_weights, output: None, dtype: std::marker::PhantomData }
    }

    pub fn forward(&mut self, input: &B) -> B {
        self.input = Some(input.clone());

        let q_weights = self.q.lock().unwrap();
        let k_weights = self.k.lock().unwrap();
        let v_weights = self.v.lock().unwrap();
        
        let query_proj = input.matmul(&q_weights.transpose());
        let key_proj = input.matmul(&k_weights.transpose());
        let value_proj = input.matmul(&v_weights.transpose());

        let raw_attention_scores = query_proj.matmul(&key_proj.transpose_axes(1, 2));
        let masked_attention_scores = raw_attention_scores.add(self.create_mask_array(input.shape()[1]));
        let normalized_attention_scores = softmax(&masked_attention_scores);

        let attention_output = normalized_attention_scores.matmul(&value_proj);
        self.output = Some(attention_output.clone());

        self.q_proj = Some(query_proj.clone());
        self.k_proj = Some(key_proj.clone());
        self.v_proj = Some(value_proj.clone());

        let projected_output = attention_output.matmul(&self.output_weights);
        self.o_proj = Some(projected_output.clone());

        projected_output
    }

    pub fn backward(&mut self, upstream_grad: &B) -> B {
        let grad_projected_output = upstream_grad.matmul(&self.o_proj.as_ref().unwrap());
    
        // Backprop through the final projection layer
        let mut grad_attention_output = grad_projected_output.matmul(&self.output_weights);

        // Gradient clipping
        let clip_value = T::from(1.0).unwrap();
        let norm = grad_attention_output.norm();
        if norm > clip_value {
            grad_attention_output = grad_attention_output.mul_scalar(clip_value / norm);
        }

        // Compute gradient of softmax
        let softmax_grad = softmax_prime(self.output.as_ref().unwrap());

        // Apply softmax gradient to the attention output gradient
        let d_attention = softmax_grad.matmul(&grad_attention_output);

        let d_v = d_attention.batch_mul(&self.v_proj.as_ref().unwrap());
        let d_k = d_attention.batch_mul(&self.k_proj.as_ref().unwrap());
        let d_q = d_attention.batch_mul(&self.q_proj.as_ref().unwrap());

        // Gradients w.r.t. to the input
        let input = self.input.as_ref().unwrap();
        let q_grad = d_q.transpose_axes(2, 1).matmul(&input);
        let k_grad = d_k.transpose_axes(2, 1).matmul(&input);
        let v_grad = d_v.transpose_axes(2, 1).matmul(&input);

        let q_weights = self.q.lock().unwrap();
        let k_weights = self.k.lock().unwrap();
        let v_weights = self.v.lock().unwrap();

        // Backpropagation to the input embedding
        let d_qkv = d_q.matmul(&q_weights.transpose())
            + d_k.matmul(&k_weights.transpose())
            + d_v.matmul(&v_weights.transpose());

        let qkv_norm = d_qkv.norm();

        if qkv_norm > clip_value {
            self.q_grad = Some(q_grad.mul_scalar(clip_value/qkv_norm));
            self.k_grad = Some(q_grad.mul_scalar(clip_value/qkv_norm));
            self.v_grad = Some(v_grad.mul_scalar(clip_value/qkv_norm));
        } else {
            self.q_grad = Some(q_grad);
            self.k_grad = Some(k_grad);
            self.v_grad = Some(v_grad);
        }

        d_qkv
    }

    pub fn update(&mut self, learning_rate: T) {
        let query_grad_mean = self.q_grad.as_ref().unwrap().mean_axis(0);
        let key_grad_mean = self.k_grad.as_ref().unwrap().mean_axis(0); 
        let value_grad_mean = self.v_grad.as_ref().unwrap().mean_axis(0);

        let t_lr = T::from(learning_rate).unwrap();

        let mut q_weights = self.q.lock().unwrap();
        let mut  k_weights = self.k.lock().unwrap();
        let mut  v_weights = self.v.lock().unwrap();

        (*q_weights) -= query_grad_mean.mul_scalar(t_lr);
        (*k_weights) -= key_grad_mean.mul_scalar(t_lr);
        (*v_weights) -= value_grad_mean.mul_scalar(t_lr);
    }

    pub fn create_mask_array(&self, seq_len: usize) -> B {
        let mask = (0..seq_len).flat_map(|i| {
            (0..seq_len).map(|j| {
                if j <= i {
                    T::from(0.0).unwrap()
                } else {
                    T::from(-f32::INFINITY).unwrap()
                }
            }).collect::<Vec<T>>()
        }).collect::<Vec<T>>();
        let mask_array = Array2::from_shape_vec((seq_len, seq_len), mask).unwrap();
        B::new(mask_array.into_dyn())
    }
}

impl<T, B> Node<T, B> for AttentionHead<T, B>
where
    T: Debug + Float + FromPrimitive + AsPrimitive<f32> + Send + Sync,
    B: Tensor<T> + Clone + AddAssign + SubAssign + Send + Sync + Add<Output = B>
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
        let mut shape = vec![3]; // Q, K, V
        let q_weights = self.q.lock().unwrap();
        shape.extend(&q_weights.shape());
        shape
    }
}

pub struct MultiHeadAttention<T,B>
where
    T: Float,
    B: Tensor<T>,
{
    d_model: usize,
    pub num_heads: usize,
    pub heads: Vec<Arc<Mutex<AttentionHead<T,B>>>>,
    pub output_weights: Arc<Mutex<B>>,
    pub output_proj: Option<B>,
    pub output_grad: Option<B>,
    dtype: std::marker::PhantomData<T>
}

impl<T,B> MultiHeadAttention<T,B>
where 
    T: Float + Debug,
    B: Tensor<T> + Clone + AddAssign + Add<Output = B> + SubAssign,
{
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        let mut heads = Vec::new();
        for _ in 0..num_heads {
            heads.push(Arc::new(Mutex::new(AttentionHead::new(d_model, num_heads))));
        }
        let output_weights = Arc::new(Mutex::new(B::glorot_uniform(&[d_model, d_model])));
        Self { d_model, num_heads, heads, output_weights, output_proj: None, output_grad: None, dtype: std::marker::PhantomData }
    }

    pub fn forward(&mut self, input: &B) -> B {
        let mut attention_outputs = Vec::new();
        let size_per_head = self.d_model / self.num_heads;

        for (i, head_mutex) in self.heads.iter_mut().enumerate() {
            let head_input = input.slice(2, i*size_per_head, (i+1)*size_per_head);
            let mut head = head_mutex.lock().unwrap();
            let attention_output = head.forward(&head_input);
            attention_outputs.push(attention_output);
        }

        let outputs_refs: Vec<&B> = attention_outputs.iter().collect();

        let head_outputs = B::concatenate(outputs_refs.as_slice(), 2);
        let output = head_outputs.matmul(&self.output_weights.lock().unwrap());

        self.output_proj = Some(output.clone());
    
        output
    }
    pub fn backward(&mut self, upstream_grad: &B) -> B {
        let output_weights = self.output_weights.lock().unwrap();
        let d_concat = upstream_grad.matmul(&output_weights.transpose());
        let last_axis = d_concat.shape().len()-1;
        let d_head_outputs: Vec<B> = d_concat.split(last_axis, self.num_heads).iter_mut()
            .map(|e| e.transpose_axes(last_axis-1, last_axis)).collect();

        let mut d_qkv_heads = Vec::new();
        for (head_mutex, d_head_output) in self.heads.iter_mut().zip(d_head_outputs) {
            let mut head = head_mutex.lock().unwrap();
            let d_qkv = head.backward(&d_head_output);
            d_qkv_heads.push(d_qkv.clone());
        }

        let grad = Tensor::concatenate(&d_qkv_heads.iter().collect::<Vec<_>>(), 2);

        let output_proj = self.output_proj.as_ref().unwrap();
        let transposed_upstream = upstream_grad.transpose_axes(2, 1);

        self.output_grad = Some(transposed_upstream.matmul(&output_proj));

        grad
    }

    pub fn update(&mut self, learning_rate: T) {
        let mean_output_grad = self.output_grad.as_ref().unwrap().mean_axis(0);
        let mut output_weights = self.output_weights.lock().unwrap();  
        (*output_weights) -= mean_output_grad.mul_scalar(learning_rate);

        for head_mutex in &mut self.heads {
            let mut head = head_mutex.lock().unwrap();
            head.update(learning_rate)
        }
    }
}

impl<T, B> Node<T, B> for MultiHeadAttention<T, B>
where
    T: Debug + Float + FromPrimitive + AsPrimitive<f32> + Send + Sync,
    B: Tensor<T> + Clone + AddAssign + SubAssign + Send + Sync + Add<Output = B>
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
        let mut shape = vec![self.num_heads];
        let head = self.heads[0].lock().unwrap();
        let q_weights = head.q.lock().unwrap();
        shape.extend(q_weights.shape());
        shape
    }
}

pub struct Block<T,B>
where
    T: Float + AsPrimitive<f32>,
    B: Tensor<T> + Clone,
{
    pub attention: Arc<Mutex<MultiHeadAttention<T,B>>>,
    pub add_norm_1: Arc<Mutex<LayerNorm<T,B>>>,
    pub feed_forward: Arc<Mutex<Sequential<T,B>>>,
    pub add_norm_2: Arc<Mutex<LayerNorm<T,B>>>,
}

impl<T,B> Block<T,B>
where
    T: Float + FromPrimitive + Debug + AsPrimitive<f32>,
    B: Tensor<T> + Clone + AddAssign + Add<Output = B> + SubAssign,
{
    pub fn new(d_model: usize, d_ff: usize, num_heads: usize) -> Self {
        let attention = Arc::new(Mutex::new(MultiHeadAttention::new(d_model, num_heads)));
        let mut feed_forward = Sequential::new();
        feed_forward.add_layer(Box::new(Linear::new(d_model, d_ff, Activation::Gelu)));
        feed_forward.add_layer(Box::new(Linear::new(d_ff, d_model, Activation::Gelu)));
        let feed_forward_mutex = Arc::new(Mutex::new(feed_forward));
        let add_norm_1 = Arc::new(Mutex::new(LayerNorm::new(d_model)));
        let add_norm_2 = Arc::new(Mutex::new(LayerNorm::new(d_model)));
        Self { attention, add_norm_1, feed_forward: feed_forward_mutex, add_norm_2 }
    }

    pub fn forward(&mut self, input: &B) -> B {
        let mut attention = self.attention.lock().unwrap();
        let mut add_norm_1 = self.add_norm_1.lock().unwrap();
        let mut feed_forward = self.feed_forward.lock().unwrap();
        let mut add_norm_2 = self.add_norm_2.lock().unwrap();

        let attention_output = attention.forward(&input);
        add_norm_1.residual = attention_output.clone();
        let add_norm_1_output = add_norm_1.forward(&attention_output);
        let feed_forward_output = feed_forward.forward(&add_norm_1_output);
        add_norm_2.residual = feed_forward_output.clone();
        let add_norm_2_output = add_norm_2.forward(&feed_forward_output);
        add_norm_2_output
    }

    pub fn backward(&mut self, upstream_grad: &B) -> B {
        let mut attention = self.attention.lock().unwrap();
        let mut add_norm_1 = self.add_norm_1.lock().unwrap();
        let mut feed_forward = self.feed_forward.lock().unwrap();
        let mut add_norm_2 = self.add_norm_2.lock().unwrap();

        let add_norm_2_output_grad = add_norm_2.backward(upstream_grad);
        let feed_forward_output_grad = feed_forward.backward(&add_norm_2_output_grad);
        let add_norm_1_output_grad = add_norm_1.backward(&feed_forward_output_grad);
        let attention_output_grad = attention.backward(&add_norm_1_output_grad);
        attention_output_grad
    }

    pub fn update(&mut self, learning_rate: T) {
        let mut attention = self.attention.lock().unwrap();
        let mut add_norm_1 = self.add_norm_1.lock().unwrap();
        let mut feed_forward = self.feed_forward.lock().unwrap();
        let mut add_norm_2 = self.add_norm_2.lock().unwrap();

        add_norm_2.update(learning_rate);
        feed_forward.update(learning_rate);
        add_norm_1.update(learning_rate);
        attention.update(learning_rate);
    }
}

impl<T, B> Node<T, B> for Block<T, B>
where
    T: Debug + Float + FromPrimitive + AsPrimitive<f32> + Send + Sync,
    B: Tensor<T> + Clone + AddAssign + SubAssign + Send + Sync + Add<Output = B>
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
        vec![0]
    }
}

pub struct PredictionOutput<T,B>
where
    T: Float + AsPrimitive<usize> + AddAssign + FromPrimitive + Debug + ScalarOperand + SampleUniform + MulAssign + SubAssign + AsPrimitive<f32> + Serialize + Deserialize<'static> + Send + Sync,
    B: 'static + Tensor<T> + Clone + AddAssign + SubAssign + Add<Output = B> + Serialize + Deserialize<'static> + Send + Sync,
{
    pub output: B,
    marker: PhantomData<T>
}

impl<T,B> PredictionOutput<T,B>
where
    T: Float + AsPrimitive<usize> + AddAssign + FromPrimitive + Debug + ScalarOperand + SampleUniform + MulAssign + SubAssign + AsPrimitive<f32> + Serialize + Deserialize<'static> + Send + Sync,
    B: 'static + Tensor<T> + Clone + AddAssign + SubAssign + Add<Output = B> + Serialize + Deserialize<'static> + Send + Sync,
{
    pub fn new() -> Self {
        Self { output: B::zeros(&[1]), marker: std::marker::PhantomData }
    }

    pub fn forward(&mut self, logits: &B) -> B {
        self.output = logits.clone();
        logits.clone()
    }

    pub fn backward(&mut self, target: &B) -> B {
        let mask = Transformer::get_mask_for_special_tokens(target, self.output.shape()[2]);
        let output_grad = Transformer::cross_entropy_loss_backward(target, &self.output, &mask);
        if validate_gradients(&output_grad) {
            println!(">>> Cross Entropy Loss backward contains NaN");
        }
        output_grad
    }

    pub fn update(&mut self, _learning_rate: T) {
        ()
    }
}

impl<T, B> Node<T, B> for PredictionOutput<T, B>
where
    T: Float + AsPrimitive<usize> + AddAssign + FromPrimitive + Debug + ScalarOperand + SampleUniform + MulAssign + SubAssign + AsPrimitive<f32> + Serialize + Deserialize<'static> + Send + Sync,
    B: 'static + Tensor<T> + Clone + AddAssign + SubAssign + Add<Output = B> + Serialize + Deserialize<'static> + Send + Sync,
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
        vec![0]
    }
}