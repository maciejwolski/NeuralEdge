#![allow(unused)]
pub mod test_data;
use test_data::DATA;

use std::collections::{HashMap, HashSet};
use std::ops::AddAssign;
use ndarray::{ArrayD, Axis};
use num_traits::{AsPrimitive, Float};

use serde::{Serialize, Deserialize};
use serde_json;
use std::fs;

#[derive(Serialize, Deserialize)]
pub struct BPETokenizer<T> 
{
    pub vocab: HashMap<String, T>,
    pub tokens: HashMap<usize, String>,
}

impl<T> BPETokenizer<T>
where T: AsPrimitive<usize> + Float + AddAssign + Serialize,
{
    pub fn new(vocab: HashMap<String, T>) -> Self {
        let tokens = vocab.iter().map(|(k, &v)| (v.as_(), k.clone())).collect();
        Self { vocab, tokens }
    }

    pub fn build_vocab(text_data: &[&str], limit: usize) -> HashMap<String, T> {
        let mut vocab = HashMap::new();
        let mut index = T::zero();

        // Insert special tokens
        let special_tokens = vec!["[START]", "[PAD]", "[EOS]", "[UNK]", "[CLS]", "[SEP]"];
        for token in special_tokens {
            vocab.insert(token.to_string(), index);
            index += T::one();
        }

        // Insert all individual characters from the dataset to ensure coverage
        let mut characters = HashSet::new();
        text_data.iter().flat_map(|s| s.chars()).for_each(|c| { characters.insert(c); });
        for c in characters {
            vocab.insert(c.to_string(), index);
            index += T::one();
        }

        // Initialize pair counts with the characters already in the vocab
        let mut pair_count = HashMap::new();
        let mut tokens: Vec<Vec<String>> = text_data.iter()
            .map(|&text| text.chars().map(|c| c.to_string()).collect::<Vec<String>>())
            .collect();

        for token_list in &tokens {
            for window in token_list.windows(2) {
                if let [a, b] = &window[..] {
                    *pair_count.entry((a.clone(), b.clone())).or_insert(0) += 1;
                }
            }
        }

        // Iteratively merge the most frequent pairs
        while vocab.len() < limit && !pair_count.is_empty() {
            let max_pair = pair_count.iter().max_by_key(|&(_, count)| count).map(|(p, _)| p.clone());
            if let Some(max_pair) = max_pair {
                let new_token = format!("{}{}", max_pair.0, max_pair.1);
                vocab.insert(new_token.clone(), index);
                index += T::one();

                // Update the tokens and pair counts
                let mut new_tokens = Vec::new();
                let mut new_pair_count = HashMap::new();

                for token_list in &tokens {
                    let mut new_token_list = Vec::new();
                    let mut i = 0;
                    while i < token_list.len() {
                        if i < token_list.len() - 1 && token_list[i] == max_pair.0 && token_list[i + 1] == max_pair.1 {
                            new_token_list.push(new_token.clone());
                            i += 2; // Skip the next element as it's merged
                        } else {
                            new_token_list.push(token_list[i].clone());
                            i += 1;
                        }
                    }
                    new_tokens.push(new_token_list);
                }

                // Recalculate pair counts
                for token_list in &new_tokens {
                    for window in token_list.windows(2) {
                        if let [a, b] = &window[..] {
                            *new_pair_count.entry((a.clone(), b.clone())).or_insert(0) += 1;
                        }
                    }
                }

                tokens = new_tokens;
                pair_count = new_pair_count;
            } else {
                break; // No more pairs to merge
            }
        }

        vocab
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current_pos = 0;
        let chars = text.chars().collect::<Vec<_>>();
        let text_length = chars.len();
    
        while current_pos < text_length {
            let mut max_length = 1; // At least one character will be taken if no matches found
            let mut token_found = None;
    
            for length in 1..=text_length - current_pos {
                let end_pos = current_pos + length;
                let candidate = chars[current_pos..end_pos].iter().collect::<String>();
                if self.vocab.contains_key(&candidate) {
                    max_length = length; // Update the length of the token to be the max found so far
                    token_found = Some(candidate);
                }
            }
    
            // Add the longest found token or the single character if no token was found
            if let Some(token) = token_found {
                tokens.push(token);
            } else {
                tokens.push(chars[current_pos].to_string());
            }
    
            current_pos += max_length;
        }
    
        tokens
    }

    pub fn pad_sequence(&self, tokens: &Vec<T>, target_len: usize) -> Vec<T> {
        let mut padded_tokens = tokens.clone();
        let pad_token_index = *self.vocab.get("[PAD]").unwrap();
    
        while padded_tokens.len() < target_len {
            padded_tokens.push(pad_token_index);
        }
    
        padded_tokens
    }

    pub fn tokenize_to_numbers(&self, text: &str) -> Vec<T> {
        let tokens = self.tokenize(text);
        let mut numbers = Vec::new();
    
        for token in tokens {
            let number = match self.vocab.get(&token) {
                Some(num) => *num,
                None => continue, // Skip tokens that are not in the vocab
            };

            numbers.push(number);
        }
    
        numbers
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn token_to_string(&self, token: T) -> String {
        for (word, &index) in &self.vocab {
            if index == token {
                return word.clone();
            }
        }
    
        "[UNK]".to_string()
    }

    pub fn tokens_to_string(&self, tokens: Vec<T>) -> String {
        let mut text = String::new();
    
        for token in tokens {
            let word = self.token_to_string(token);
            text.push_str(&word);
        }
    
        text
    }

    pub fn get_dataset_token_size(&self, dataset: &Vec<&str>) -> usize {
        let mut token_size = 0;
        for text in dataset {
            let tokens = self.tokenize(text);
            token_size += tokens.len();
        }
    
        token_size
    }

    pub fn get_ordered_tokens(&self) -> Vec<String> {
        let mut vocab: Vec<(&String, &T)> = self.vocab.iter().collect();
        // Use `sort_by` and manually handle floating-point comparisons
        vocab.sort_by(|a, b| {
            a.1.partial_cmp(b.1).expect("Attempted to compare a NaN or other non-comparable value")
        });
        let vocab: Vec<String> = vocab.into_iter().map(|(k, _)| k.clone()).collect();
    
        vocab
    }

    pub fn save_tokenizer(&self) {
        // Convert to a vector for sorting
        let mut vec: Vec<(&String, &T)> = self.vocab.iter().collect();
        // Sort the vector by values
        vec.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal));

        let vocab_json = serde_json::to_string_pretty(&vec).expect("Error serializing Tokenizer [vocab]");
        fs::write("tokenizer_vocab.json", vocab_json).expect("Error while saving tokenizer JSON");
    }
}

pub fn load_tokenizer<T: for<'de> Deserialize<'de>>() -> BPETokenizer<T> 
where
    T: Float + AddAssign + AsPrimitive<usize> + Serialize,
{
    let data: String = fs::read_to_string("tokenizer_vocab.json").expect("Error while loading tokenizer JSON");

    let vec: Vec<(String, T)> = serde_json::from_str(&data).expect("Error while deserializing Tokenizer JSON");

    // Convert the vector of tuples back into a HashMap
    let vocab: HashMap<String, T> = vec.into_iter().collect();

    BPETokenizer::new(vocab)
}

pub fn get_first_pad_index_in_batch<T: Float>(vocab: &Vec<String>, tokens: &ArrayD<T>) -> usize {
    let pad_index = T::from(vocab.iter().position(|s| s == "[PAD]").unwrap()).unwrap();

    for (i, row) in tokens.axis_iter(Axis(1)).enumerate() {
        if row.iter().any(|&x| x == pad_index) {
            return i;
        }
    }

    // Default return value when there are no elements to iterate on
    0
}