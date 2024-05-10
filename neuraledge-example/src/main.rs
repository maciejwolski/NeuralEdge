extern crate neuraledge_lang;

use neuraledge_core::backends::{Tensor, cpu::CpuTensor};
use neuraledge_core::nn::optimizers::RMSProp;

use neuraledge_lang::transformers::Transformer;
use neuraledge_lang::tokenizers::{BPETokenizer, test_data::DATA};
use neuraledge_lang::transformers::train::*;

use ndarray::Array;
use std::path::Path;

use neuraledge_lang::tokenizers::load_tokenizer;

fn main() {
    let data = DATA.to_vec();

    let tokenizer: BPETokenizer::<f32>;

    if Path::new("tokenizer_vocab.json").exists(){
        tokenizer = load_tokenizer();
        println!("Tokenizer loaded from file...")
    } else {
        tokenizer = BPETokenizer::<f32>::new(BPETokenizer::build_vocab(&data, 300));
        tokenizer.save_tokenizer();
    }
    
    let vocab: Vec<String> = tokenizer.get_ordered_tokens();

    let (train_data, target_data) = split_data_target(&tokenizer, &data.iter().map(|s| s.to_string()).collect());

    let train_data_str: Vec<&str> = train_data.iter().map(|s| s.as_str()).collect();
    let target_data_str: Vec<&str> = target_data.iter().map(|s| s.as_str()).collect();

    let mut transformer: Transformer::<f32, CpuTensor<f32>>;
        
    if Path::new("model.bin").exists() {
        let data = std::fs::read_to_string("model.bin").expect("Failed to read model from file");
        transformer = Transformer::<f32, CpuTensor<f32>>::load_model(data);
        println!("Model loaded from file");
    } else {
        transformer = Transformer::<f32, CpuTensor<f32>>::new(vocab.len(), 8, 16, 1, 2);
    }

    let optimizer = RMSProp::<f32, CpuTensor<f32>>::default();

    println!("Training model...");
    println!("{:?} parameters, {:?} tokens in training dataset", transformer.get_model_size(), tokenizer.get_dataset_token_size(&train_data_str));

    let (train_data, train_target_data, test_data, test_target_data) = train_test_split(&train_data_str, &target_data_str, 0.2);

    println!("vocab size: {} tokens", vocab.len());

    train(&mut transformer, &train_data, &train_target_data, &test_data[0], test_target_data[0], &tokenizer, &vocab, 24, 1000, 0.01, Some(optimizer));

    println!("Testing model...");

    for i in 0..test_data.len() {
        let tokenized_input = tokenizer.tokenize_to_numbers(&test_data[i]);
        let tokenized_target = tokenizer.tokenize_to_numbers(&test_target_data[i]);

        let tokenized_target_len = tokenized_target.len();

        let padded_input = tokenizer.pad_sequence(&tokenized_input, tokenized_target_len);

        println!("Input: {:?} / Target: {:?}", padded_input, tokenized_target);
        let input_array = Array::from_shape_vec((1, tokenized_target_len), padded_input).unwrap();
        let target_array = Array::from_shape_vec((1, tokenized_target_len), tokenized_target).unwrap();

        println!("input array: {:?}", input_array);

        let mut input = CpuTensor::new(input_array.into_dyn());
        let mut target = CpuTensor::new(target_array.into_dyn());

        let output = transformer.generate_text(&mut input, &vocab, tokenized_target_len, true);

        println!("target array: {:?}", target.get_data());

        let loss = transformer.cross_entropy_loss(&mut target);
        let avg_loss = loss / tokenized_target_len as f32;
        println!("Input text: {}", test_data[i]);
        println!("Target text: {}", test_target_data[i]);
        println!("Output: {:?}, Loss: {}", output, avg_loss);
        println!("---------------------------");
    }
}