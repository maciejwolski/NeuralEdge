pub mod components;
pub mod train;
pub mod utils;

use utils::*;

use crate::neuraledge_core::backends::Tensor;
use crate::neuraledge_core::nn::linear::{Linear, Sequential};
use crate::neuraledge_core::nn::optimizers::RMSProp;
use crate::neuraledge_core::nn::activations::Activation;
use crate::neuraledge_core::backends::graph::Graph;
use crate::tokenizers::get_first_pad_index_in_batch;

use components::{Embedding, RotaryPositionalEncoding, Block};

use std::{
    collections::HashMap,
    fmt::Debug,
    ops::{Add, AddAssign, SubAssign, MulAssign},
    sync::{Arc, Mutex},
};
use num_traits::{AsPrimitive, Float, FromPrimitive};
use rand::distributions::uniform::SampleUniform;

use serde::{Serialize, Deserialize};

use ndarray::{s, Array, Axis, ScalarOperand};

use self::components::{MultiHeadAttention, PredictionOutput};

pub struct Transformer<T,B>
where
    T: Float + AsPrimitive<usize> + AddAssign + FromPrimitive + Debug + ScalarOperand + SampleUniform + MulAssign + SubAssign + AsPrimitive<f32> + Serialize + Deserialize<'static> + Send + Sync,
    B: 'static + Tensor<T> + Clone + AddAssign + SubAssign + Add<Output = B> + Serialize + Deserialize<'static> + Send + Sync,
{
    params: Parameters<T, B>,
    tok_embedding: Arc<Mutex<Embedding<T, B>>>,
    pos_encoding: RotaryPositionalEncoding<T,B>,
    layers: Vec<Block<T,B>>,
    fc_logits: Arc<Mutex<Linear<T, B>>>,
    prediction: Arc<Mutex<PredictionOutput<T,B>>>,
    output: Option<B>,
    opt_grads: HashMap::<String, B>,
    hyperparams: Hyperparameters,
    graph: Graph<T,B>
}

impl<T,B> Transformer<T,B>
where
    T: Float + AsPrimitive<usize> + AddAssign + FromPrimitive + Debug + ScalarOperand + SampleUniform + MulAssign + SubAssign + AsPrimitive<f32> + Serialize + Deserialize<'static> + Send + Sync,
    B: 'static + Tensor<T> + Clone + AddAssign + SubAssign + Add<Output = B> + Serialize + Deserialize<'static> + Send + Sync,
{
    pub fn new(vocab_size: usize, d_model: usize, d_ff: usize, num_layers: usize, num_heads: usize) -> Self {
        let mut graph = Graph::new();

        let pos_encoding = RotaryPositionalEncoding::new(d_model);
        let tok_embedding = Arc::new(Mutex::new(Embedding::new(vocab_size, d_model, pos_encoding.clone())));
        let tok_embed = Arc::clone(&tok_embedding);

        graph.add_layer(tok_embed);
        
        let mut layers: Vec<Block<T, B>> = Vec::new();
        for _ in 0..num_layers {
            let block = Block::new(d_model, d_ff, num_heads);

            let attention = Arc::clone(&block.attention);
            graph.add_layer(attention);

            let add_norm_1 = Arc::clone(&block.add_norm_1);
            graph.add_layer(add_norm_1);

            let sequential = block.feed_forward.lock().unwrap();
            let feed_forward1 = Arc::clone(&sequential.layers[0]);
            let feed_forward2 = Arc::clone(&sequential.layers[1]);
            graph.add_layer(feed_forward1);
            graph.add_layer(feed_forward2);

            drop(sequential);

            let add_norm_2 = Arc::clone(&block.add_norm_2);
            graph.add_layer(add_norm_2);

            layers.push(block);
        }

        let fc_logits = Arc::new(Mutex::new(Linear::new(d_model, vocab_size, Activation::Softmax)));
        let logits = Arc::clone(&fc_logits);
        graph.add_layer(logits);

        let prediction = Arc::new(Mutex::new(PredictionOutput::<T,B>::new()));
        let prediction_output = Arc::clone(&prediction);
        graph.add_layer(prediction_output);

        let hyperparams = Hyperparameters { 
            learning_rate: 0.0, 
            num_epochs: 0,
            batch_size: 0,
            num_layers,
            num_heads,
            d_model,
            d_ff,
        };

        Self { params: Parameters::new(), tok_embedding, pos_encoding, layers, fc_logits, prediction, output: None, opt_grads: HashMap::new(), hyperparams, graph }
    }

    fn from_checkpoint(model: ModelData<T,B>) -> Self {
        let tensor_fields = model.tensors;
        let hyperparams = model.hyperparameters;

        let mut transformer = Transformer::new(0, hyperparams.d_model, hyperparams.d_ff, hyperparams.num_layers, hyperparams.num_heads);
        transformer.hyperparams.num_epochs = hyperparams.num_epochs;

        // fill parameters struct
        transformer.params.tok_embedding_weights = Arc::new(Mutex::new(tensor_fields.tok_embedding_weights));
        transformer.params.fc_logits_weights = Arc::new(Mutex::new(tensor_fields.fc_logits_weights));
        transformer.params.fc_bias = Arc::new(Mutex::new(tensor_fields.fc_biases));
        
        for i in 0..tensor_fields.attention_head_query.len() {
            transformer.params.attention_head_query.push(Arc::new(Mutex::new(tensor_fields.attention_head_query[i].to_owned())));
            transformer.params.attention_head_key.push(Arc::new(Mutex::new(tensor_fields.attention_head_key[i].to_owned())));
            transformer.params.attention_head_value.push(Arc::new(Mutex::new(tensor_fields.attention_head_value[i].to_owned())));
        }

        for i in 0..tensor_fields.ff_layer_weights.len() {
            transformer.params.ff_layer_weights.push(Arc::new(Mutex::new(tensor_fields.ff_layer_weights[i].to_owned())));
            transformer.params.ff_layer_biases.push(Arc::new(Mutex::new(tensor_fields.ff_biases[i].to_owned())));
        }

        for i in 0..tensor_fields.layer_norms_weights.len() {
            transformer.params.layer_norms_weights.push(Arc::new(Mutex::new(tensor_fields.layer_norms_weights[i].to_owned())));
        }

        for i in 0..tensor_fields.layer_norms_biases.len() {
            transformer.params.layer_norms_biases.push(Arc::new(Mutex::new(tensor_fields.layer_norms_biases[i].to_owned())));
        }

        // fill transformer struct

        transformer.tok_embedding = Arc::new(Mutex::new(
            Embedding::from_checkpoint(transformer.params.tok_embedding_weights.lock().unwrap().to_owned(), hyperparams.d_model, transformer.pos_encoding.clone())
        ));
        transformer.fc_logits = Arc::new(Mutex::new(
            Linear::from_checkpoint(transformer.params.fc_logits_weights.lock().unwrap().to_owned(), transformer.params.fc_bias.lock().unwrap().to_owned(), Activation::Softmax)
        ));

        // fill blocks (hyperparams.num_layers)
        let mut ff_counter = 0;
        let mut norm_counter = 0;

        for i in 0..transformer.layers.len() {
            let mut block = Block::new(hyperparams.d_model, hyperparams.d_ff, hyperparams.num_heads);
            block.attention = Arc::new(Mutex::new(MultiHeadAttention::new(hyperparams.d_model, hyperparams.num_heads)));
            let mut sequential = Sequential::<T, B>::new();
            sequential.add_layer(Box::new(Linear::new(hyperparams.d_model, hyperparams.d_ff, Activation::Gelu)));
            sequential.add_layer(Box::new(Linear::new(hyperparams.d_ff, hyperparams.d_model, Activation::Gelu)));
            block.feed_forward = Arc::new(Mutex::new(sequential));

            for h in 0..hyperparams.num_heads {
                let mh_attn = block.attention.lock().unwrap();
                let mut head = mh_attn.heads[h].lock().unwrap();
                head.q = Arc::clone(&transformer.params.attention_head_query[i*h+h]);
                head.k = Arc::clone(&transformer.params.attention_head_key[i*h+h]);
                head.v = Arc::clone(&transformer.params.attention_head_value[i*h+h]);
            }

            let mut bff = 0;
            for l in ff_counter..ff_counter+2 {
                block.feed_forward.lock().unwrap().layers[bff].lock().unwrap().weights = Arc::clone(&transformer.params.ff_layer_weights[l]);
                block.feed_forward.lock().unwrap().layers[bff].lock().unwrap().bias = Arc::clone(&transformer.params.ff_layer_biases[l]);
                ff_counter += 1;
                bff += 1;
            }

            block.add_norm_1.lock().unwrap().weights = Arc::clone(&transformer.params.layer_norms_weights[norm_counter]);
            block.add_norm_1.lock().unwrap().bias = Arc::clone(&transformer.params.layer_norms_biases[norm_counter]);
            block.add_norm_2.lock().unwrap().weights = Arc::clone(&transformer.params.layer_norms_weights[norm_counter+1]);
            block.add_norm_2.lock().unwrap().bias = Arc::clone(&transformer.params.layer_norms_biases[norm_counter+1]);
            norm_counter += 2;

        }

        transformer
    }

    pub fn forward(&mut self, result: &B) -> B {
        // let mut embed = self.tok_embedding.lock().unwrap();
        // let encoded_batch = embed.forward(input);

        // let mut output = encoded_batch;
        // for layer in &mut self.layers {
        //     output = layer.forward(&output);
        // }


        // let mut logits = self.fc_logits.lock().unwrap();
        // let result = logits.forward(&output);
        self.output = Some(result.clone());
        result.clone()
    }

    pub fn backward(&mut self, target: &B) -> B {
        let mask = Transformer::get_mask_for_special_tokens(target, self.output.as_ref().unwrap().shape()[2]);
        let output_grad = Transformer::cross_entropy_loss_backward(target, &self.output.as_ref().unwrap(), &mask);
        if validate_gradients(&output_grad) {
            println!(">>> Cross Entropy Loss backward contains NaN");
        }
        // self.output_grad = Some(output_grad.clone());

        // let mut logits = self.fc_logits.lock().unwrap();

        // let mut grad = output_grad.clone();
        // grad = logits.backward(&grad);

        // for layer in self.layers.iter_mut().rev() {
        //     grad = layer.backward(&grad);
        // }

        // let mut embed = self.tok_embedding.lock().unwrap();
        // embed.backward(&grad);
        output_grad
    }

    pub fn update(&mut self, learning_rate: T, opt: Option<&mut RMSProp<T,B>>) {

        if opt.is_some() {

            self.collect_gradients();

            let optimizer = opt.unwrap();

            optimizer.step(&mut self.opt_grads);

        } else {

            self.graph.update(learning_rate);
            
        }
    }

    pub fn cross_entropy_loss(&mut self, target: &mut B) -> T {
        let output = &self.prediction.lock().unwrap().output; // output of the last ff layer

        let t_eps = T::from(1e-10).unwrap();

        let mut loss = T::zero();

        for i in 0..target.shape()[0] {
            for j in 0..target.shape()[1] {
                let target_index: usize = target.get_data()[[i, j]].as_();
                let output_prob = output.get_data()[[i, j, target_index]] + t_eps;
                loss += -output_prob.ln();
            }
        }
        loss / T::from(output.shape()[0] * output.shape()[1]).unwrap()
    }

    pub fn cross_entropy_loss_backward(target: &B, output: &B, mask: &B) -> B {
        //let output = self.output.as_ref().unwrap();
        let target_onehot = B::onehot(target, output.shape()[2]);

        //let mask = self.get_mask_for_special_tokens(target, output.shape()[2]);
        let output_grad = output.sub(&target_onehot);
        let masked_output_grad = output_grad.mul(&mask);

        masked_output_grad
    }

    pub fn get_mask_for_special_tokens(target: &B, num_classes: usize) -> B {
        let mut mask = B::ones(&[target.shape()[0], target.shape()[1], num_classes]);
        let zero_vector = Array::from_vec(vec![T::from(0.0).unwrap(); num_classes]);

        for i in 0..target.shape()[0] {
            for j in 0..target.shape()[1] {
                let target_index = target.get_data()[[i, j]].to_usize().unwrap();

                // mask out special tokens
                if target_index < 6 {
                    mask.get_data_mut().slice_mut(s![i,j,..]).assign(&zero_vector);
                }
            }
        }

        mask
    }

    pub fn generate_text(&mut self, input: &mut B, vocab: &Vec<String>, max_length: usize, show_predicted: bool) -> Vec<String> {
        let seq_start = get_first_pad_index_in_batch(&vocab, &input.get_data());

        let mut texts = vec![String::new(); input.shape()[0]];
        let mut mut_input = input.clone();
        let mut output: B;

        for i in 0..input.shape()[0] {
            for j in 0..seq_start{
                let index: usize = input.get_data()[[i, j]].as_();
                let word = &vocab[index];
                texts[i].push_str(word);
            }
        }

        for i in seq_start..max_length {

            // forward pass
            //output = self.forward(&mut_input.clone());
            output = self.graph.forward(&mut_input);

            // self.prediction.lock().unwrap().output = output.clone();

            // get current output and assign to self.output
            // let last_output_slice = output.get_data().index_axis(Axis(1), i);
            // let mut prediction = self.prediction.lock().unwrap();
            // prediction.output.get_data_mut().slice_mut(s![.., i..i+1, ..]).assign(&last_output_slice.insert_axis(Axis(1)));

            // get max values - predicted tokens
            let index = output.index_axis_move(1, i).max_axis(1);
            let predicted = index.get_data().to_owned();
            let word_indices = predicted.iter().map(|&x| x.as_()).collect::<Vec<usize>>();

            let predicted_tensor = B::new(Array::from(predicted).insert_axis(Axis(1)).into_dyn());

            // d_model, seq_len
            mut_input.get_data_mut().slice_mut(s![.., i..i+1]).assign(&predicted_tensor.view());

            for i in 0..texts.len() {
                let word = &vocab[word_indices[i] as usize];
                texts[i].push_str(word);
            }
        }

        if show_predicted {
            println!("predicted array: {:?}", mut_input.get_data());
        }

        texts
    }

    pub fn get_prediction_output(&self) -> B {
        let prediction = self.prediction.lock().unwrap();
        prediction.output.clone()
    }
    
    pub fn get_model_size(&self) -> usize {
        let embed_size = {
            let tok_embed = self.tok_embedding.lock().unwrap();
            let size = tok_embed.weights.lock().unwrap().size();
            size
        };
    
        let attn_out_size = {
            let attn = self.layers[0].attention.lock().unwrap();
            let size = attn.output_weights.lock().unwrap().size();
            size
        };
    
        let (q_head_size, num_heads) = {
            let attn = self.layers[0].attention.lock().unwrap();
            let head = attn.heads[0].lock().unwrap();
            let q_head = head.q.lock().unwrap();
            (q_head.size(), attn.num_heads)
        };
    
        let (ff1_weight_size, ff2_weight_size) = {
            let sequential = self.layers[0].feed_forward.lock().unwrap();
            let ff1_weights = sequential.layers[0].lock().unwrap().weights.lock().unwrap().size();
            let ff2_weights = sequential.layers[1].lock().unwrap().weights.lock().unwrap().size();
            (ff1_weights, ff2_weights)
        };
    
        let (ln1_weight_size, ln2_weight_size) = {
            let ln1 = self.layers[0].add_norm_1.lock().unwrap();
            let ln2 = self.layers[0].add_norm_2.lock().unwrap();
            let size1 = ln1.weights.lock().unwrap().size();
            let size2 = ln2.weights.lock().unwrap().size();
            (size1, size2)
        };
    
        let logits_weight_size = {
            let logits = self.fc_logits.lock().unwrap();
            let size = logits.weights.lock().unwrap().size();
            size
        };
    
        let mut size = embed_size;
        for _ in &self.layers {
            // Each Q, K, V size multiplied by 3 for each head
            size += q_head_size * 3 * num_heads;
            size += attn_out_size;
            size += ff1_weight_size;
            size += ff2_weight_size;
            size += ln1_weight_size;
            size += ln2_weight_size;
        }
        size += logits_weight_size;
    
        size
    }

    pub fn collect_gradients(&mut self) {
        
        let fc_logits = self.fc_logits.lock().unwrap();
        if validate_gradients(&fc_logits.weights_grad) {
            println!(">>> Gradients for fc_logits weights are NaN");
        } else {
            let key = "fc_logits_weights";
            if let Some(value) = self.opt_grads.get_mut(key) {
                *value = fc_logits.weights_grad.clone();
            } else {
                self.opt_grads.insert(key.to_string(), fc_logits.weights_grad.clone());
            }
        }

        let mut n = self.layers.len()-1;
        for layer in &mut self.layers.iter_mut().rev() {

            let add_norm_1 = layer.add_norm_1.lock().unwrap();

            if validate_gradients(&add_norm_1.weights_grad) {
                println!(">>> Gradients for layer_norms_weights_{} are NaN", 1);
            } else {
                let key = format!("layer_norms_weights_{}", n*2);
                if let Some(value) = self.opt_grads.get_mut(&key) {
                    *value = add_norm_1.weights_grad.clone();
                } else {
                    self.opt_grads.insert(key, add_norm_1.weights_grad.clone());
                }
            }

            let add_norm_2 = layer.add_norm_2.lock().unwrap();

            if validate_gradients(&add_norm_2.weights_grad) {
                println!(">>> Gradients for layer_norms_weights_{} are NaN", 2);
            } else {
                let key = format!("layer_norms_weights_{}", n*2+1);
                if let Some(value) = self.opt_grads.get_mut(&key) {
                    *value = add_norm_2.weights_grad.clone();
                } else {
                    self.opt_grads.insert(key, add_norm_2.weights_grad.clone());
                }
            }

            if validate_gradients(&add_norm_1.bias_grad) {
                println!(">>> Gradients for layer_norms_biases_{} are NaN", 1);
            } else {
                let key = format!("layer_norms_biases_{}", n*2);
                if let Some(value) = self.opt_grads.get_mut(&key) {
                    *value = add_norm_1.bias_grad.clone();
                } else {
                    self.opt_grads.insert(key, add_norm_1.bias_grad.clone());
                }
            }

            if validate_gradients(&add_norm_2.bias_grad) {
                println!(">>> Gradients for layer_norms_biases_{} are NaN", 2);
            } else {
                let key = format!("layer_norms_biases_{}", n*2+1);
                if let Some(value) = self.opt_grads.get_mut(&key) {
                    *value = add_norm_2.bias_grad.clone();
                } else {
                    self.opt_grads.insert(key, add_norm_2.bias_grad.clone());
                }
            }

            let feed_forward = layer.feed_forward.lock().unwrap();

            if validate_gradients(&feed_forward.layers[0].lock().unwrap().weights_grad) {
                println!(">>> Gradients for ff_layer_weights_{} are NaN", 0);
            } else {
                let key0 = format!("ff_layer_weights_{}", n*2);
                if let Some(value) = self.opt_grads.get_mut(&key0) {
                    *value = feed_forward.layers[0].lock().unwrap().weights_grad.clone();
                } else {
                    self.opt_grads.insert(key0, feed_forward.layers[0].lock().unwrap().weights_grad.clone());
                }
            }
            
            if validate_gradients(&feed_forward.layers[1].lock().unwrap().weights_grad) {
                println!(">>> Gradients for ff_layer_weights_{} are NaN", 1);
            } else {
                let key1 = format!("ff_layer_weights_{}", n*2+1);
                if let Some(value) = self.opt_grads.get_mut(&key1) {
                    *value = feed_forward.layers[1].lock().unwrap().weights_grad.clone();
                } else {
                    self.opt_grads.insert(key1, feed_forward.layers[1].lock().unwrap().weights_grad.clone());
                }
            }

            if validate_gradients(&feed_forward.layers[0].lock().unwrap().bias_grad) {
                println!(">>> Gradients for ff_layer_biases_{} are NaN", 0);
            } else {
                let key0 = format!("ff_layer_biases_{}", n*2);
                if let Some(value) = self.opt_grads.get_mut(&key0) {
                    *value = feed_forward.layers[0].lock().unwrap().bias_grad.clone();
                } else {
                    self.opt_grads.insert(key0, feed_forward.layers[0].lock().unwrap().bias_grad.clone());
                }
            }

            if validate_gradients(&feed_forward.layers[1].lock().unwrap().bias_grad) {
                println!(">>> Gradients for ff_layer_biases_{} are NaN", 1);
            } else {
                let key1 = format!("ff_layer_biases_{}", n*2+1);
                if let Some(value) = self.opt_grads.get_mut(&key1) {
                    *value = feed_forward.layers[1].lock().unwrap().bias_grad.clone();
                } else {
                    self.opt_grads.insert(key1, feed_forward.layers[1].lock().unwrap().bias_grad.clone());
                }
            }

            let mut h_counter = 0;
            for head_mutex in &layer.attention.lock().unwrap().heads {
                let head = head_mutex.lock().unwrap();
                if validate_gradients(&head.q_grad.clone().unwrap()) {
                    println!(">>> Gradients for attention_head_query_{} are NaN", h_counter);
                } else {
                    let key = format!("attention_head_query_{}", h_counter);
                    if let Some(value) = self.opt_grads.get_mut(&key) {
                        *value = head.q_grad.clone().unwrap();
                    } else {
                        self.opt_grads.insert(key, head.q_grad.clone().unwrap());
                    }
                }
            
                if validate_gradients(&head.k_grad.clone().unwrap()) {
                    println!(">>> Gradients for attention_head_key_{} are NaN", h_counter);
                } else {
                    let key = format!("attention_head_key_{}", h_counter);
                    if let Some(value) = self.opt_grads.get_mut(&key) {
                        *value = head.k_grad.clone().unwrap();
                    } else {
                        self.opt_grads.insert(key, head.k_grad.clone().unwrap());
                    }
                }
            
                if validate_gradients(&head.v_grad.clone().unwrap()) {
                    println!(">>> Gradients for attention_head_value_{} are NaN", h_counter);
                } else {
                    let key = format!("attention_head_value_{}", h_counter);
                    if let Some(value) = self.opt_grads.get_mut(&key) {
                        *value = head.v_grad.clone().unwrap();
                    } else {
                        self.opt_grads.insert(key, head.v_grad.clone().unwrap());
                    }
                }
    
                // if validate_gradients(&head.o_grad.clone().unwrap()) {
                //     println!(">>> Gradients for attention_output_{} are NaN", h_counter);
                // } else {
                //     let key = format!("attention_output_{}", h_counter);
                //     if let Some(value) = self.opt_grads.get_mut(&key) {
                //         *value = head.o_grad.clone().unwrap();
                //     } else {
                //         self.opt_grads.insert(key, head.o_grad.clone().unwrap());
                //     }
                // }

                h_counter += 1;
            }
            
            let attention = layer.attention.lock().unwrap();
            let attn_output_grad = attention.output_grad.as_ref().unwrap();

            if validate_gradients(attn_output_grad) {
                println!(">>> Gradients for attention_output_{} are NaN", n);
            } else {
                let key = format!("attention_output_{}", n);
                if let Some(value) = self.opt_grads.get_mut(&key) {
                    *value = attn_output_grad.clone();
                } else {
                    self.opt_grads.insert(key, attn_output_grad.clone());
                }
            }

            if n > 0 {
                n -= 1;
            }
        }

        let embed = self.tok_embedding.lock().unwrap();
        if validate_gradients(&embed.grads.clone().unwrap()) {
            println!(">>> Gradients for fc_logits weights are NaN");
        } else {
            let key = "tok_embedding";
            if let Some(value) = self.opt_grads.get_mut(key) {
                *value = embed.grads.clone().unwrap();
            } else {
                self.opt_grads.insert(key.to_string(), embed.grads.clone().unwrap());
            }
        }
    }

    pub fn register_parameters(&mut self, optimizer: &mut RMSProp<T, B>) {
        self.params = Parameters::new();
        let mut parameters_to_register: Vec<(String, Arc<Mutex<B>>)> = Vec::new();

        self.params.tok_embedding_weights = Arc::clone(&self.tok_embedding.lock().unwrap().weights);

        let mut n = 0;
        for layer in &mut self.layers {

            for head_mutex in &layer.attention.lock().unwrap().heads {
                let head = head_mutex.lock().unwrap();
                self.params.attention_head_query.push(Arc::clone(&head.q));
                self.params.attention_head_key.push(Arc::clone(&head.k));
                self.params.attention_head_value.push(Arc::clone(&head.v));
            }
                
            self.params.attention_outputs.push(Arc::clone(&layer.attention.lock().unwrap().output_weights));

            let sequential_layers = &mut layer.feed_forward.lock().unwrap().layers;
            self.params.ff_layer_weights.push(Arc::clone(&sequential_layers[0].lock().unwrap().weights));
            self.params.ff_layer_biases.push(Arc::clone(&sequential_layers[0].lock().unwrap().bias));
            self.params.ff_layer_weights.push(Arc::clone(&sequential_layers[1].lock().unwrap().weights));
            self.params.ff_layer_biases.push(Arc::clone(&sequential_layers[1].lock().unwrap().bias));
            
            self.params.layer_norms_weights.push(Arc::clone(&layer.add_norm_1.lock().unwrap().weights));
            self.params.layer_norms_weights.push(Arc::clone(&layer.add_norm_2.lock().unwrap().weights));
            self.params.layer_norms_biases.push(Arc::clone(&layer.add_norm_1.lock().unwrap().bias));
            self.params.layer_norms_biases.push(Arc::clone(&layer.add_norm_2.lock().unwrap().bias));

            if n > 0{
                n += 1;
            }
        }

        self.params.fc_logits_weights = Arc::clone(&self.fc_logits.lock().unwrap().weights);
        self.params.fc_bias = Arc::clone(&self.fc_logits.lock().unwrap().bias);

        for field in self.params.clone().into_iter() {
            parameters_to_register.push(field);
        }

        for component in parameters_to_register {
            let (name, param) = component;
            optimizer.register_parameters(name, param);
        }
    }

    pub fn save_model(&self, path: &str) {
        let model_data = ModelData {
            tensors: self.into(),
            hyperparameters: self.hyperparams.clone(),
            dtype: std::marker::PhantomData,
            marker: std::marker::PhantomData,
        };

        let mut saveguard = true;

        let mut i = 0;
        for tensor in model_data.tensors.into_iter() {
            let tensor = tensor.clone();
            let check = validate_saved_component(&tensor, model_data.tensors.names[i].clone());
            i += 1;

            if !check {
                saveguard = false;
            }
        }

        // save only if the data is not corrupted by NaNs
        if saveguard == true {
            let serialized = serde_json::to_string(&model_data).unwrap();
            std::fs::write(path, serialized).expect("Failed to write model to file");
        }
    }

    pub fn load_model(data: String) -> Transformer<T, B> {
        let static_data: &'static str = Box::leak(data.into_boxed_str());
        let model_data: ModelData<'static, T, B> = serde_json::from_str(static_data).unwrap();
        Transformer::from_checkpoint(model_data)
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuraledge_core::backends::cpu::CpuTensor;
    use crate::tokenizers::*;

    #[test]
    fn test_transformer_graph() {
        let d_model = 8;
        let d_ff = 16;
        let num_layers = 1;
        let num_heads = 2;
        let tokenizer = BPETokenizer::new(BPETokenizer::build_vocab(&crate::tokenizers::test_data::DATA.to_vec(), 300));
        let vocab: Vec<String> = tokenizer.get_ordered_tokens();
        let vocab_size = vocab.len();
        
        let data = ["Fortune favors "];
        let batch_size = data.len();

        let mut transformer = Transformer::<f32, CpuTensor<f32>>::new(vocab_size, d_model, d_ff, num_layers, num_heads);
        
        let mut input_tokens = tokenizer.tokenize_to_numbers(data[0]);

        let target_data = ["Fortune favors the bold."];
        let target_tokens = tokenizer.tokenize_to_numbers(target_data[0]);
        let target_len = target_tokens.len();

        let padded_input = tokenizer.pad_sequence(input_tokens.as_mut(), target_len);

        let mut input = CpuTensor::new(Array::from_vec(padded_input).into_shape((batch_size, target_len)).unwrap().into_dyn());
        let target = CpuTensor::new(Array::from_vec(target_tokens).into_shape((batch_size, target_len)).unwrap().into_dyn());

        // Generate text (& Graph FORWARD)
        transformer.generate_text(&mut input, &vocab, target_len, false);

        // Graph BACKWARD
        transformer.graph.backward(&target);

        // Graph UPDATE
        transformer.graph.update(0.1);
        
    }
}