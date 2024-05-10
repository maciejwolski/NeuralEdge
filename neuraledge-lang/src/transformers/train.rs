use crate::neuraledge_core::nn::optimizers::RMSProp;
use crate::tokenizers::BPETokenizer;
use super::*;

use rand::seq::SliceRandom;
use rand::thread_rng;

use std::time::Instant;
use std::fmt::Debug;
use std::iter::Sum;


pub fn train<T, B>(transformer: &mut Transformer<T,B>, train_data: &Vec<&str>, target_data: &Vec<&str>, test: &str, target_test: &str, tokenizer: &BPETokenizer<T>, vocab: &Vec<String>, batch_size: usize, num_epochs: usize, lr: f32, mut opt: Option<RMSProp<T,B>>) 
where
    T: Float + AddAssign + AsPrimitive<usize> + FromPrimitive + Debug + MulAssign + ScalarOperand + SampleUniform + SubAssign + AsPrimitive<f32> + Serialize + Deserialize<'static> + Sum + Send + Sync,
    B: Tensor<T> + Clone + SubAssign + AddAssign + Add<Output = B> + Serialize + Deserialize<'static> + Send + Sync + 'static,
{
    let mut epoch_losses = Vec::<T>::new();

    transformer.hyperparams.learning_rate = lr;
    transformer.hyperparams.batch_size = batch_size;

    let mut lr = T::from(lr).unwrap();
    let mut indices: Vec<_> = (0..train_data.len()).collect();

    let start_epoch = transformer.hyperparams.num_epochs;

    for epoch in start_epoch..start_epoch+num_epochs as u32 {
        indices.shuffle(&mut thread_rng());
        let start = Instant::now();

        let mut total_loss = T::from(0.0).unwrap();
        let mut total_tokens = 0;

        // register parameters for optimization on the first epoch
        if epoch == start_epoch && opt.is_some() {
            opt.as_mut().unwrap().learning_rate = lr;
            transformer.register_parameters(&mut opt.as_mut().unwrap());
        }

        for i in (0..indices.len()-batch_size).step_by(batch_size) {

            let mut input_tokens_batch = Vec::new();
            let mut target_tokens_batch = Vec::new();

            let mut max_length = 0;
            let mut max_target_length = 0;

            for j in i..i+batch_size {
                let idx = indices[j];
                let input_tokens = tokenizer.tokenize_to_numbers(train_data[idx]);
                let input_len = input_tokens.len();
                if input_len > max_length {
                    max_length = input_len;
                }
                input_tokens_batch.push(input_tokens);

                let target_tokens = tokenizer.tokenize_to_numbers(target_data[idx]);
                let target_len = target_tokens.len();
                if target_len > max_target_length {
                    max_target_length = target_len;
                }
                target_tokens_batch.push(target_tokens);
            }

            for input_sequence in input_tokens_batch.iter_mut() {
                let padded_input = tokenizer.pad_sequence(input_sequence, max_target_length);
                *input_sequence = padded_input;
            }

            for target_sequence in target_tokens_batch.iter_mut() {
                let padded_target = tokenizer.pad_sequence(target_sequence, max_target_length);
                *target_sequence = padded_target;
            }

            let input_tokens_array = Array::from_shape_vec((batch_size, max_target_length), input_tokens_batch.into_iter().flatten().collect()).unwrap();
            let target_tokens_array = Array::from_shape_vec((batch_size, max_target_length), target_tokens_batch.into_iter().flatten().collect()).unwrap();

            let mut input = B::new(input_tokens_array.into_dyn());
            let mut target = B::new(target_tokens_array.into_dyn());

            transformer.generate_text(&mut input, vocab, max_target_length as usize, false);
            let loss = transformer.cross_entropy_loss(&mut target);

            total_loss += loss;
            total_tokens += max_target_length;

            // warmup
            let warmup_period = 20;
            let tuned_lr: T;
            if epoch < warmup_period {
                tuned_lr = T::from(epoch-start_epoch).unwrap() * lr / T::from(warmup_period).unwrap();
            } else {
                lr = lr * T::from(0.999).unwrap();
                tuned_lr = lr
            }

            transformer.graph.backward(&target);
            transformer.update(tuned_lr, opt.as_mut());

        }
        epoch_losses.push(total_loss / T::from(total_tokens).unwrap());

        println!("Epoch: {}, Loss: {:?}, Time: {:?}", epoch, total_loss / T::from(total_tokens).unwrap(), start.elapsed());

        transformer.hyperparams.num_epochs = epoch as u32;

        // save model if the loss is lower than the average of previous epochs
        if epoch_losses.len() > 5 {
            let last_loss = epoch_losses.last().unwrap();
            let avg_loss = epoch_losses.iter().skip(epoch_losses.len()-5).map(|&x| x).sum::<T>() / T::from(5.0).unwrap();
        
            if last_loss < &avg_loss {
                transformer.save_model("model.bin");
            }
        }

        // evaluate the progress in test generation
        if epoch % 50 == 0 {
            let tokenized_input = tokenizer.tokenize_to_numbers(test);
            let tokenized_target = tokenizer.tokenize_to_numbers(target_test);

            let tokenized_target_len = tokenized_target.len();

            let padded_input = tokenizer.pad_sequence(&tokenized_input, tokenized_target_len);

            println!("---------------------------");
            println!("Input: {:?} / Target: {:?}", padded_input, tokenized_target);
            let input_array = Array::from_shape_vec((1, tokenized_target_len), padded_input).unwrap();
            let target_array = Array::from_shape_vec((1, tokenized_target_len), tokenized_target).unwrap();

            println!("input array: {:?}", input_array);

            let mut input = B::new(input_array.into_dyn());
            let mut target = B::new(target_array.into_dyn());

            let output = transformer.generate_text(&mut input, &vocab, tokenized_target_len, true);

            println!("target array: {:?}", target.get_data());

            let loss = transformer.cross_entropy_loss(&mut target);
            let avg_loss = loss / T::from(tokenized_target_len).unwrap();
            println!("Input text: {}", test);
            println!("Target text: {}", target_test);
            println!("Output: {:?}, Validation Loss: {:?}", output, avg_loss);
            println!("---------------------------");
        }
    }
}

pub fn train_test_split<'a>(data: &'a Vec<&str>, target_data: &'a Vec<&str>, test_size: f64) -> (Vec<&'a str>, Vec<&'a str>, Vec<&'a str>, Vec<&'a str>) {
    let test_size = (data.len() as f64 * test_size).round() as usize;
    let train_size = data.len() - test_size;

    let train_data = data[..train_size].to_vec();
    let train_target_data = target_data[..train_size].to_vec();
    let test_data = data[train_size..].to_vec();
    let test_target_data = target_data[train_size..].to_vec();

    (train_data, train_target_data, test_data, test_target_data)
}

pub fn split_data_target<T>(tokenizer: &BPETokenizer<T>, data: &Vec<String>) -> (Vec<String>, Vec<String>)
    where T: AsPrimitive<usize> + Float + AddAssign + Serialize
{
    let mut train_data = Vec::new();
    let mut target_data = Vec::new();

    for string in data {
        let tokens = tokenizer.tokenize(string);
        let split_index = tokens.len() * 2 / 3;

        let data_part_tokens = &tokens[..split_index];
        let data_part = data_part_tokens.join("");
        let target_part = tokens.join("");

        train_data.push(data_part);
        target_data.push(target_part);
    }

    (train_data, target_data)
}