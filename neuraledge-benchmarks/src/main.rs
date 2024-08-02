extern crate neuraledge_core;
use neuraledge_core::backends::{Tensor, cpu::CpuTensor, gpu::GpuTensor, gpu::Block};
use ndarray::{ArrayBase, ArrayD, Dim, IxDyn, IxDynImpl, OwnedRepr};
use rand::Rng;

use std::time::Instant;

/*
    General comments to consider:
    - CPU handles vectors better
    - GPU handles contiguous arrays better
    - IxDyn is much slower than fixed size arrays

*/

fn main() {
    neuraledge_core::backends::gpu::init_gpu_context().wait();
    let mut rng = rand::thread_rng();

    let mut arrays: Vec<ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>> = Vec::new();

    for _ in 0..5000 {
        let data = ArrayD::<f32>::from_shape_fn(IxDyn(&[50, 384]), |_| {
            rng.gen_range(-1.0..1.0)
        });

        arrays.push(data);
    }

    let mut cpu_tensors = Vec::new();
    let mut gpu_tensors = Vec::new();

    for array in &arrays {
        cpu_tensors.push(CpuTensor::new(array.clone()));
        gpu_tensors.push(GpuTensor::new(array.clone()));
    }

    for iteration in 1..5 {
        println!("Iteration #{}", iteration);

        println!("Benchmarking CPU processing...");
        let start = Instant::now();

        let cpu_batch = CpuTensor::stack(&cpu_tensors.iter().collect::<Vec<_>>(), 0);
        cpu_batch.matmul(&cpu_tensors[0].transpose());

        let cpu_elapsed = start.elapsed();
        println!("Time: {:?}", cpu_elapsed);

        let mult = GpuTensor::new_on_gpu(arrays[0].clone(), None);
        let batch = GpuTensor::<f32>::stack(gpu_tensors.iter().collect::<Vec<_>>().as_slice(), 0);

        println!("Benchmarking GPU stacked processing...");
        let start = Instant::now();

        batch.matmul(&mult);

        let gpu_stacked_elapsed = start.elapsed();
        println!("Total time: {:?}", gpu_stacked_elapsed);

        println!("====================");
    }
}
