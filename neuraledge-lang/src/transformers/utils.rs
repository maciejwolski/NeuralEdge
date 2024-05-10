use super::*;
use serde::{Serialize, Deserialize};

pub fn validate_gradients<T, B: Tensor<T>>(grad: &B) -> bool 
where 
    T: Float + Clone,
{
    // check if there are any NaN values in the gradients
    let nan_count = grad.get_data().iter().filter(|&x| x.is_nan()).count();
    let total_count = grad.get_data().len();
    let nan_ratio = nan_count as f32 / total_count as f32;
    if nan_ratio > 0.0 {
        println!("NaN ratio: {} ", nan_ratio);
        true
    } else {
        false
    }
}

pub fn validate_saved_component<T, B: Tensor<T>>(tensor: &B, name: String) -> bool
where 
    T: Float + Clone + Add<Output = T>,
{
    let nan_count = tensor.get_data().iter().filter(|&x| x.is_nan()).count();
    let total_count = tensor.get_data().len();
    let nan_ratio = nan_count as f32 / total_count as f32;
    if nan_ratio > 0.0 {
        println!("NaN ratio for {}: {} ", name, nan_ratio);
        return false
    }
    true
}

#[derive(Clone)]
pub struct Parameters<T, B: Tensor<T>> 
where 
    T: Float + Clone,
{
    pub tok_embedding_weights: Arc<Mutex<B>>,
    pub attention_head_query: Vec<Arc<Mutex<B>>>,
    pub attention_head_key: Vec<Arc<Mutex<B>>>,
    pub attention_head_value: Vec<Arc<Mutex<B>>>,
    pub attention_outputs: Vec<Arc<Mutex<B>>>,
    pub mh_outputs: Vec<Arc<Mutex<B>>>,
    pub ff_layer_weights: Vec<Arc<Mutex<B>>>,
    pub ff_layer_biases: Vec<Arc<Mutex<B>>>,
    pub fc_logits_weights: Arc<Mutex<B>>,
    pub fc_bias: Arc<Mutex<B>>,
    pub layer_norms_weights: Vec<Arc<Mutex<B>>>,
    pub layer_norms_biases: Vec<Arc<Mutex<B>>>,
    dtype: std::marker::PhantomData<T>,
}

impl<T: 'static, B: Tensor<T>> Parameters<T, B> 
where 
    T: Float + Clone,
{
    pub fn new() -> Self {
        Self {
            tok_embedding_weights: Arc::new(Mutex::new(B::zeros(&[0,0]))),
            attention_head_query: Vec::new(),
            attention_head_key: Vec::new(),
            attention_head_value: Vec::new(),
            attention_outputs: Vec::new(),
            mh_outputs: Vec::new(),
            ff_layer_weights: Vec::new(),
            ff_layer_biases: Vec::new(),
            fc_logits_weights: Arc::new(Mutex::new(B::zeros(&[0,0]))),
            fc_bias: Arc::new(Mutex::new(B::zeros(&[0,0]))),
            layer_norms_weights: Vec::new(),
            layer_norms_biases: Vec::new(),
            dtype: std::marker::PhantomData,
        }
    }

    pub fn into_iter(self) -> Vec<(String, Arc<Mutex<B>>)> {
        let mut parameters: Vec<(String, Arc<Mutex<B>>)> = Vec::new();

        parameters.push(("tok_embedding".to_string(), self.tok_embedding_weights));

        let mut hq = 0;
        for tensor in self.attention_head_query {
            parameters.push((format!("attention_head_query_{}", hq).to_string(), tensor));
            hq += 1;
        }

        let mut hk = 0;
        for tensor in self.attention_head_key {
            parameters.push((format!("attention_head_key_{}", hk).to_string(), tensor));
            hk += 1;
        }

        let mut hv = 0;
        for tensor in self.attention_head_value {
            parameters.push((format!("attention_head_value_{}", hv).to_string(), tensor));
            hv += 1;
        }

        let mut ao = 0;
        for tensor in self.attention_outputs {
            parameters.push((format!("attention_output_{}", ao).to_string(), tensor));
            ao += 1;
        }

        let mut ff = 0;
        for tensor in self.ff_layer_weights {
            parameters.push((format!("ff_layer_weights_{}", ff).to_string(), tensor));
            ff += 1;
        }

        let mut fb = 0;
        for tensor in self.ff_layer_biases {
            parameters.push((format!("ff_layer_biases_{}", fb).to_string(), tensor));
            fb += 1;
        }

        let mut lnw = 0;
        for tensor in self.layer_norms_weights {
            parameters.push((format!("layer_norms_weights_{}", lnw).to_string(), tensor));
            lnw += 1;
        }

        let mut lnb = 0;
        for tensor in self.layer_norms_biases {
            parameters.push((format!("layer_norms_biases_{}", lnb).to_string(), tensor));
            lnb += 1;
        }

        parameters.push(("fc_logits_weights".to_string(), self.fc_logits_weights));
        parameters.push(("fc_bias".to_string(), self.fc_bias));

        parameters
    }
}

#[derive(Serialize, Deserialize)]
pub struct ModelData<'a, T, B: Tensor<T>> 
where 
    T: Float + Clone,
    B: Clone + AddAssign + SubAssign,
{
    pub tensors: SerializableTensors<'a,T,B>,
    pub hyperparameters: Hyperparameters,
    pub dtype: std::marker::PhantomData<T>,
    pub marker: std::marker::PhantomData<&'a ()>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Hyperparameters {
    pub learning_rate: f32,
    pub num_epochs: u32,
    pub batch_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub d_model: usize,
    pub d_ff: usize,
}

#[derive(Serialize, Deserialize)]
pub struct SerializableTensors<'a, T, B: Tensor<T>> 
where 
    T: Float + Clone,
    B: Clone + AddAssign + SubAssign
{
    pub names: Vec<String>,
    pub tok_embedding_weights: B,
    pub attention_head_query: Vec<B>,
    pub attention_head_key: Vec<B>,
    pub attention_head_value: Vec<B>,
    pub attention_outputs: Vec<B>,
    pub ff_layer_weights: Vec<B>,
    pub ff_biases: Vec<B>,
    pub layer_norms_weights: Vec<B>,
    pub layer_norms_biases: Vec<B>,
    pub fc_logits_weights: B,
    pub fc_biases: B,
    dtype: std::marker::PhantomData<T>,
    marker: std::marker::PhantomData<&'a ()>,
}

impl<'a, T: 'static, B: Tensor<T>> From<&Transformer<T,B>> for SerializableTensors<'a, T,B> 
where 
    T: Float + AsPrimitive<usize> + AddAssign + FromPrimitive + Debug + ScalarOperand + SampleUniform + MulAssign + SubAssign + AsPrimitive<f32> + Serialize + Deserialize<'static> + Send + Sync,
    B: 'static + Tensor<T> + Clone + AddAssign + SubAssign + Add<Output = B> + Serialize + Deserialize<'static> + Send + Sync,
{
    fn from(model: &Transformer<T,B>) -> Self {
        let mut attention_head_query = Vec::new();
        let mut attention_head_key = Vec::new();
        let mut attention_head_value = Vec::new();
        let mut attention_outputs = Vec::new();
        let mut ff_layer_weights = Vec::new();
        let mut ff_biases = Vec::new();
        let mut layer_norms_weights = Vec::new();
        let mut layer_norms_biases = Vec::new();

        let mut comp_names = Vec::new();
        comp_names.push("tok_embedding_embeddings".to_string());

        // Generate component names
        for i in 0..model.layers.len() {
            for head_id in 0..model.hyperparams.num_heads {
                comp_names.push(format!("attention_head_{}_query", head_id));
                comp_names.push(format!("attention_head_{}_key", head_id));
                comp_names.push(format!("attention_head_{}_value", head_id));
                comp_names.push(format!("attention_output_{}", head_id));
            }
            for _ in 0..model.layers[i].feed_forward.lock().unwrap().layers.len() {
                comp_names.push(format!("ff_layer_{}_weights", i));
                comp_names.push(format!("ff_layer_{}_bias", i));
            }
            comp_names.push(format!("layer_norms_weights{}_1", i));
            comp_names.push(format!("layer_norms_weights{}_2", i));
            comp_names.push(format!("layer_norms_biases{}_1", i));
            comp_names.push(format!("layer_norms_biases{}_2", i));
        }
        comp_names.push("fc_logits_weights".to_string());
        comp_names.push("fc_logits_bias".to_string());

        // Collect all weights and biases data
        for layer in &model.layers {
            let attention_heads = layer.attention.lock().unwrap().heads.clone();
            for head in attention_heads {
                let head = head.lock().unwrap();
                attention_head_query.push(head.q.lock().unwrap().clone());
                attention_head_key.push(head.k.lock().unwrap().clone());
                attention_head_value.push(head.v.lock().unwrap().clone());
            }
            attention_outputs.push(layer.attention.lock().unwrap().output_weights.lock().unwrap().clone());
            
            let sequential = layer.feed_forward.lock().unwrap();
            let ff_layers = sequential.layers.clone();
            for ff_layer in ff_layers {
                let ff = ff_layer.lock().unwrap();
                ff_layer_weights.push(ff.weights.lock().unwrap().clone());
                ff_biases.push(ff.bias.lock().unwrap().clone());
            }

            let ln1 = layer.add_norm_1.lock().unwrap();
            let ln2 = layer.add_norm_2.lock().unwrap();
            layer_norms_weights.push(ln1.weights.lock().unwrap().clone());
            layer_norms_weights.push(ln2.weights.lock().unwrap().clone());
            layer_norms_biases.push(ln1.bias.lock().unwrap().clone());
            layer_norms_biases.push(ln2.bias.lock().unwrap().clone());
        }

        let tok_embedding_weights = model.tok_embedding.lock().unwrap().weights.lock().unwrap().clone();
        let fc_logits = model.fc_logits.lock().unwrap();
        let fc_logits_weights = fc_logits.weights.lock().unwrap().clone();
        let fc_biases = fc_logits.bias.lock().unwrap().clone();

        Self {
            names: comp_names,
            tok_embedding_weights,
            attention_head_query,
            attention_head_key,
            attention_head_value,
            attention_outputs,
            ff_layer_weights,
            ff_biases,
            layer_norms_weights,
            layer_norms_biases,
            fc_logits_weights,
            fc_biases,
            dtype: std::marker::PhantomData,
            marker: std::marker::PhantomData,
        }
    }
}

pub struct SerializableTensorsIter<'a, T, B: Tensor<T>> 
where 
    T: Float + Clone + Add<Output = T>,
    B: Clone + AddAssign + SubAssign
{
    tensors: &'a SerializableTensors<'a, T,B>,
    index: usize,
    state: usize
}

impl<'a, T, B: Tensor<T>> Iterator for SerializableTensorsIter<'a, T, B> 
where 
    T: Float + Clone + Add<Output = T>,
    B: Clone + AddAssign + SubAssign,
{
    type Item = &'a B;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.state {
                0 => {
                    self.state += 1;
                    return Some(&self.tensors.tok_embedding_weights);
                },
                1 => {
                    if let Some(item) = self.tensors.attention_head_query.get(self.index) {
                        self.index += 1;
                        return Some(item);
                    } else {
                        self.state += 1;
                        self.index = 0;
                    }
                },
                2 => {
                    if let Some(item) = self.tensors.attention_head_key.get(self.index) {
                        self.index += 1;
                        return Some(item);
                    } else {
                        self.state += 1;
                        self.index = 0;
                    }
                },
                3 => {
                    if let Some(item) = self.tensors.attention_head_value.get(self.index) {
                        self.index += 1;
                        return Some(item);
                    } else {
                        self.state += 1;
                        self.index = 0;
                    }
                },
                4 => {
                    if let Some(item) = self.tensors.attention_outputs.get(self.index) {
                        self.index += 1;
                        return Some(item);
                    } else {
                        self.state += 1;
                        self.index = 0;
                    }
                },
                5 => {
                    if let Some(item) = self.tensors.ff_layer_weights.get(self.index) {
                        self.index += 1;
                        return Some(item);
                    } else {
                        self.state += 1;
                        self.index = 0;
                    }
                },
                6 => {
                    if let Some(item) = self.tensors.ff_biases.get(self.index) {
                        self.index += 1;
                        return Some(item);
                    } else {
                        self.state += 1;
                        self.index = 0;
                    }
                },
                7 => {
                    if let Some(item) = self.tensors.layer_norms_weights.get(self.index) {
                        self.index += 1;
                        return Some(item);
                    } else {
                        self.state += 1;
                        self.index = 0;
                    }
                },
                8 => {
                    if let Some(item) = self.tensors.layer_norms_biases.get(self.index) {
                        self.index += 1;
                        return Some(item);
                    } else {
                        self.state += 1;
                        self.index = 0;
                    }
                },
                9 => {
                    self.state += 1;
                    return Some(&self.tensors.fc_logits_weights);
                },
                10 => {
                    self.state += 1;
                    return Some(&self.tensors.fc_biases);
                },
                _ => return None,
            }
        }
    }
}

impl<'a, T, B: Tensor<T>> IntoIterator for &'a SerializableTensors<'a, T, B> 
where 
    T: Float + Clone + Add<Output = T>,
    B: Clone + AddAssign + SubAssign,
{
    type Item = &'a B;
    type IntoIter = SerializableTensorsIter<'a, T, B>;

    fn into_iter(self) -> Self::IntoIter {
        SerializableTensorsIter {
            tensors: self,
            index: 0,
            state: 0,
        }
    }
}