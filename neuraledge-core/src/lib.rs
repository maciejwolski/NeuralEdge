pub mod backends;
pub mod nn;

#[cfg(test)]
mod tests {
    use crate::backends::graph::Graph;
    use crate::backends::cpu::CpuTensor;
    use crate::backends::Tensor;
    use crate::nn::linear::Linear;
    use crate::nn::activations::Activation;

    use std::sync::{Arc, Mutex};

    #[test]
    fn test_sequential_graph() {
        let layer1 = Arc::new(Mutex::new(Linear::<f32, CpuTensor<f32>>::new(32, 64, Activation::Sigmoid)));
        let layer2 = Arc::new(Mutex::new(Linear::<f32, CpuTensor<f32>>::new(64, 32, Activation::Sigmoid)));

        let mut graph = Graph::<f32, CpuTensor<f32>>::new();
        graph.add_layer(layer1);
        graph.add_layer(layer2);

        let input = CpuTensor::<f32>::random(&[1, 3, 32]);
        let output_grad = CpuTensor::<f32>::new(ndarray::Array::from_elem((1, 32), 0.1).into_dyn());

        graph.forward(&input);
        graph.backward(&output_grad);
        graph.update(0.1);
    }
}