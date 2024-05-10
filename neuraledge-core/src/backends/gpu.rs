use crate::backends::Tensor;

use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD, IxDyn};
use wgpu::{Adapter, Buffer, Device, DeviceDescriptor, Queue};

pub struct GpuResources {
    pub device: Device,
    pub queue: Queue,
}

impl GpuResources {
    pub fn new(adapter: &Adapter) -> Self {
        let (device, queue) = adapter.request_device(&DeviceDescriptor::default(), None);
        
        Self {
            device,
            queue,
        }
    }
}

const gpu_resources: GpuResources = GpuResources::new(&Adapter::request_device().unwrap());

pub struct GpuTensor<T> {
    pub data: Buffer,
    pub shape: Vec<usize>,
    pub dtype: std::marker::PhantomData<T>,
}

impl<T> Tensor<T> for GpuTensor<T> {
    
    fn new(data: ArrayD<T>) -> Self {
        let buffer_data = data.as_slice().expect("Failed to get buffer data");
        let buffer = gpu_resources.device.create_buffer(BufferDescriptor { label: (), size: (), usage: (), mapped_at_creation: () });
        
        Self {
            data: buffer,
            shape: data.shape().to_vec(),
            dtype: std::marker::PhantomData,
        }
    }

    fn view(&self) -> ArrayViewD<T> {

    }

    fn get_data(&self) -> &ArrayD<T> {

    }

    fn get_data_mut(&mut self) -> &mut ArrayD<T> {

    }

    fn set_data(&mut self, data: ArrayD<T>) {

    }

    fn reshape(&mut self, shape: &[usize]) {

    }

    fn transpose(&self) -> Self {

    }

    fn transpose_axes(&self, axis_1: usize, axis_2: usize) -> Self {

    }

    fn insert_axis(&mut self, axis: usize) {

    }

    fn stack(tensors: &[&Self], axis: usize) -> Self {

    }

    fn concatenate(tensors: &[&Self], axis: usize) -> Self {

    }

    fn split(&self, axis: usize, num_parts: usize) -> Vec<Self> {

    }

    fn matmul(&self, other: &Self) -> Self {

    }

    fn batch_mul(&self, other: &Self) -> Self {

    }

    fn mul(&self, other: &Self) -> Self {

    }

    fn div(&self, other: &Self) -> Self {

    }

    fn add(&self, other: &Self) -> Self {

    }

    fn sub(&self, other: &Self) -> Self {

    }

    fn mul_scalar(&self, scalar: T) -> Self {

    }

    fn div_scalar(&self, scalar: T) -> Self {

    }

    fn add_scalar(&self, scalar: T) -> Self {

    }

    fn sub_scalar(&self, scalar: T) -> Self {

    }

    fn zeros(shape: &[usize]) -> Self {

    }

    fn zeros_like(tensor: &Self) -> Self {

    }

    fn ones(shape: &[usize]) -> Self {

    }

    fn max_axis(&self, axis: usize) -> Self {

    }

    fn mean_axis(&self, axis: usize) -> Self {

    }

    fn sum_axis(&self, axis: usize) -> Self {

    }

    fn add_axis(&self, axis: usize, values: &Array1<T>) -> Self {

    }

    fn sub_axis(&self, axis: usize, values: &Array1<T>) -> Self {

    }

    fn norm(&self) -> T {

    }

    fn variance(&self, axis: usize) -> Self {

    }

    fn sqrt(&self) -> Self {

    }

    fn pow(&self, exp: T) -> Self {

    }

    fn glorot_uniform(shape: &[usize]) -> Self {

    }

    fn he_uniform(shape: &[usize]) -> Self {

    }

    fn random(shape: &[usize]) -> Self {

    }

    fn slice(&self, axis: usize, start: usize, end: usize) -> Self {

    }

    fn slice_mut(&mut self, axis: usize, start: usize, end: usize) -> ArrayViewMutD<T> {

    }

    fn index_axis(&self, axis: usize, index: usize) -> ArrayViewD<T> {

    }

    fn index_axis_move(&self, axis: usize, index: usize) -> Self {

    }

    fn iter(&self) -> ndarray::iter::Iter<'_, T, IxDyn> {

    }

    fn iter_mut(&mut self) -> ndarray::iter::IterMut<'_, T, IxDyn> {

    }

    fn shape(&self) -> Vec<usize> {

    }

    fn size(&self) -> usize {

    }

    fn raw_dim(&self) -> IxDyn {

    }

    fn onehot(target: &Self, num_classes: usize) -> Self {

    }

}
