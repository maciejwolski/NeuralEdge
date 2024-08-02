use super::Tensor;
use std::collections::HashMap;
use std::sync::{Arc, RwLock, OnceLock};
use std::ops::{AddAssign, SubAssign, Div, Sub, Add};
use ndarray::{Axis, ArrayD, ArrayViewD, ArrayViewMutD, IxDyn, Array, Array1};
use rand::distributions::{Uniform, Distribution};
use rand::Rng;
use serde::{Deserialize, Deserializer};
use wgpu::util::DeviceExt;

use serde::{Serialize, Serializer, ser::SerializeStruct};
use tokio::sync::OnceCell;

type WorkgroupSize = (usize, usize);
type MemoryKey = (String, WorkgroupSize);

static GPU_CONTEXT: OnceLock<OnceCell<GPUContext>> = OnceLock::new();

fn ensure_init() -> &'static OnceCell<GPUContext> {
    GPU_CONTEXT.get_or_init(OnceCell::new)
}

pub async fn init_gpu_context() -> &'static GPUContext {
    ensure_init().get_or_init(GPUContext::new).await
}

pub fn get_gpu_context() -> &'static GPUContext {
    GPU_CONTEXT.get().and_then(|cell| cell.get()).unwrap()
}

pub struct GPUContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pipelines: RwLock<HashMap<MemoryKey, Arc<wgpu::ComputePipeline>>>
}

impl GPUContext {

    pub const ARRAYOPS: &'static str = include_str!("shaders/arrayops.wgsl");
    pub const SCALAROPS: &'static str = include_str!("shaders/scalarops.wgsl");
    pub const MATMUL: &'static str = include_str!("shaders/matmul.wgsl");

    pub async fn new() -> Self {
        let (device, queue) = Self::get_device().await.unwrap();
        Self { device, queue, pipelines: RwLock::new(HashMap::new()) }
    }

    async fn get_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        //let instance = wgpu::Instance::new(wgpu::InstanceDescriptor{backends: wgpu::Backends::VULKAN, ..Default::default()});

        let request_adapter_options = wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance, // HighPerformance/LowPower
            compatible_surface: None,
            force_fallback_adapter: false,
        };

        let adapter = instance
            .request_adapter(&request_adapter_options)
            .await
            .expect("Failed to find an appropriate adapter!");

        //println!("Using GPU: {}", adapter.get_info().name);
        let gpu_limits = adapter.limits();

        let dq = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: gpu_limits,
                },
                None,
            )
            .await
            .unwrap();
        
        Some(dq)
    }

    pub fn get_pipeline(&self, label: &'static str, shader_type: &'static str, workgroup_size: WorkgroupSize) -> Arc<wgpu::ComputePipeline> {

        let mut pipelines = self.pipelines.write().unwrap();
        let key = (label.to_string(), workgroup_size);

        if let Some(pipeline) = pipelines.get(&key) {
            return pipeline.clone();
        }

        let shader_source = match shader_type {
            "array" => Self::ARRAYOPS,
            "scalar" => Self::SCALAROPS,
            "matmul" => Self::MATMUL,
            _ => panic!("Invalid shader type")
        };

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            }),
            entry_point: label,
            compilation_options: wgpu::PipelineCompilationOptions { constants: &HashMap::new(), zero_initialize_workgroup_memory: true },
        });

        pipelines.insert(key.clone(), Arc::new(pipeline));

        Arc::clone(pipelines.get(&key).unwrap())
    }
}

pub struct GpuTensor<'a, T> {
    pub data: ArrayD<T>,
    pub buffer: wgpu::Buffer,
    context: &'a GPUContext
}

impl<T> Tensor<T> for GpuTensor<'_, T>
where 
    T: num_traits::Float + bytemuck::Pod + 'static + num_traits::Zero + num_traits::One + num_traits::FromPrimitive + std::cmp::PartialOrd + Div<Output = T> + Sub<Output = T>
    + std::iter::Sum + rand::distributions::uniform::SampleUniform,
{
    fn new(data: ArrayD<T>) -> Self {
        GpuTensor::new_on_gpu(data, None)
    }

    fn view(&self) -> ArrayViewD<T> {
        self.data.view()
    }

    fn get_data(&self) -> &ArrayD<T> {
        &self.data
    }

    fn get_data_mut(&mut self) -> &mut ArrayD<T> {
        &mut self.data
    }

    fn set_data(&mut self, data: ArrayD<T>) {
        self.data = data;
    }

    fn reshape(&mut self, shape: &[usize]) {
        self.set_data(self.data.clone().into_shape(shape).unwrap())
    }

    fn transpose(&self) -> Self {
        let transposed = match self.data.ndim() {
            2 => self.data.view().permuted_axes(IxDyn(&[1, 0])).to_owned(),
            3 => self.data.view().permuted_axes(IxDyn(&[1, 0, 2])).to_owned(),
            _ => panic!("Unsupported number of dimensions!"),
        };
        GpuTensor::new_on_gpu(transposed, None)
    }

    fn transpose_axes(&self, axis_1: usize, axis_2: usize) -> Self {
        let mut data = self.data.clone();
        data.swap_axes(axis_1, axis_2);

        GpuTensor::new_on_gpu(data, None)
    }

    fn insert_axis(&mut self, axis: usize) {
        self.set_data(self.data.clone().insert_axis(Axis(axis)));
    }

    fn stack(tensors: &[&Self], axis: usize) -> Self {
        let stacked_data = tensors
            .iter()
            .map(|tensor| tensor.data.view())
            .collect::<Vec<ArrayViewD<T>>>();

        GpuTensor::new_on_gpu(ndarray::stack(Axis(axis), &stacked_data).unwrap(), None)
    }

    fn concatenate(tensors: &[&Self], axis: usize) -> Self {
        let concatenated_data = tensors
            .iter()
            .map(|tensor| tensor.data.view())
            .collect::<Vec<ArrayViewD<T>>>();

        GpuTensor::new_on_gpu(ndarray::concatenate(Axis(axis), &concatenated_data).unwrap(), None)
    }

    fn split(&self, axis: usize, num_parts: usize) -> Vec<Self> {
        let axis_size = self.data.shape()[axis];
        let step = (axis_size as f32 / num_parts as f32).ceil();
        
        let mut result = Vec::new();
        for i in 0..num_parts {
            let start = i * step as usize;
            let end = (i + 1) * step as usize;
            let split_data = self.data.slice_axis(Axis(axis), (start..end).into()).to_owned();
            result.push(GpuTensor::new_on_gpu(split_data, None));
        }

        result
    }

    fn matmul(&self, other: &Self) -> Self {
        let context = get_gpu_context();
        let target_shape: Vec<usize>;
        let wsize: usize;
        if other.shape().len() == 2 {
            target_shape = vec![self.shape()[0], self.shape()[1], other.shape()[1]];
            wsize = target_shape[0] * target_shape[1] * target_shape[2];
        } else if self.shape()[1] == other.shape()[2] {
            target_shape = vec![self.shape()[0], self.shape()[1], other.shape()[2]];
            wsize = target_shape[0] * target_shape[1] * target_shape[2];
        } else {
            target_shape = vec![self.shape()[0], self.shape()[1], other.shape()[2]];
            wsize = target_shape[0] * target_shape[1] * target_shape[2];
        }
        let pipeline: Arc<wgpu::ComputePipeline> = context.get_pipeline("batch_mul", "matmul", (32, 32));

        let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (wsize * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false
        });

        let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (wsize * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let bind_group = other.get_bind_group_batch(&self, &pipeline, &output_buffer);

        let result = other.encode_and_submit_batch(&pipeline, &bind_group, &output_buffer, &staging_buffer, &target_shape, wsize).wait();
        
        let output = result.map(|&e| T::from(e).unwrap());
        
        GpuTensor::new_on_gpu(output, None)
    }

    fn batch_mul(&self, other: &Self) -> Self {
        // placeholder
        GpuTensor::matmul(&self, other)
    }

    fn mul(&self, other: &Self) -> Self {
        // placeholder
        GpuTensor::matmul(&self, other)
    }

    fn div(&self, other: &Self) -> Self {
        let context = get_gpu_context();
        let wsize = self.size();
        let pipeline = context.get_pipeline("div", "array", (wsize, 1));

        let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (wsize * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false
        });

        let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (wsize * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let bind_group = self.get_bind_group_zip(&other, &pipeline, &output_buffer);
        let result = self.encode_and_submit(&pipeline, &bind_group, &output_buffer, &staging_buffer, &self.shape()).wait();
        let output = result.map(|&e| T::from(e).unwrap());
        
        GpuTensor::new_on_gpu(output, None)
    }

    fn add(&self, other: &Self) -> Self {
        let context = get_gpu_context();
        let wsize = self.size();
        let pipeline = context.get_pipeline("add", "array", (wsize, 1));

        let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (wsize * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false
        });

        let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (wsize * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let bind_group = self.get_bind_group_zip(&other, &pipeline, &output_buffer);
        let result = self.encode_and_submit(&pipeline, &bind_group, &output_buffer, &staging_buffer, &self.shape()).wait();
        let output = result.map(|&e| T::from(e).unwrap());
        
        GpuTensor::new_on_gpu(output, None)
    }

    fn sub(&self, other: &Self) -> Self {
        let context = get_gpu_context();
        let wsize = self.size();
        let pipeline = context.get_pipeline("sub", "array", (wsize, 1));

        let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (wsize * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false
        });

        let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (wsize * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let bind_group = self.get_bind_group_zip(&other, &pipeline, &output_buffer);
        let result = self.encode_and_submit(&pipeline, &bind_group, &output_buffer, &staging_buffer, &self.shape()).wait();
        let output = result.map(|&e| T::from(e).unwrap());
        
        GpuTensor::new_on_gpu(output, None)
    }

    fn mul_scalar(&self, scalar: T) -> Self {
        let context = get_gpu_context();
        let wsize = self.size();
        let pipeline = context.get_pipeline("mul_scalar", "scalar", (wsize, 1));

        let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (wsize * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false
        });

        let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (wsize * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let scalar_data = ArrayD::from_elem(IxDyn(&[1]), scalar);
        let scalar = GpuTensor::<T>::new_on_gpu(scalar_data, None);

        let bind_group = self.get_bind_group_zip(&scalar, &pipeline, &output_buffer);
        let result = self.encode_and_submit(&pipeline, &bind_group, &output_buffer, &staging_buffer, &self.shape()).wait();
        let output = result.map(|&e| T::from(e).unwrap());
        
        GpuTensor::new_on_gpu(output, None)
    }

    fn div_scalar(&self, scalar: T) -> Self {
        let context = get_gpu_context();
        let wsize = self.size();
        let pipeline = context.get_pipeline("div_scalar", "scalar", (wsize, 1));

        let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (wsize * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false
        });

        let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (wsize * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let scalar_data = ArrayD::from_elem(IxDyn(&[1]), scalar);
        let scalar = GpuTensor::<T>::new_on_gpu(scalar_data, None);

        let bind_group = self.get_bind_group_zip(&scalar, &pipeline, &output_buffer);
        let result = self.encode_and_submit(&pipeline, &bind_group, &output_buffer, &staging_buffer, &self.shape()).wait();
        let output = result.map(|&e| T::from(e).unwrap());
        
        GpuTensor::new_on_gpu(output, None)
    }

    fn add_scalar(&self, scalar: T) -> Self {
        let context = get_gpu_context();
        let wsize = self.size();
        let pipeline = context.get_pipeline("add_scalar", "scalar", (wsize, 1));

        let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (wsize * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false
        });

        let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (wsize * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let scalar_data = ArrayD::from_elem(IxDyn(&[1]), scalar);
        let scalar = GpuTensor::<T>::new_on_gpu(scalar_data, None);

        let bind_group = self.get_bind_group_zip(&scalar, &pipeline, &output_buffer);
        let result = self.encode_and_submit(&pipeline, &bind_group, &output_buffer, &staging_buffer, &self.shape()).wait();
        let output = result.map(|&e| T::from(e).unwrap());
        
        GpuTensor::new_on_gpu(output, None)
    }

    fn sub_scalar(&self, scalar: T) -> Self {
        let context = get_gpu_context();
        let wsize = self.size();
        let pipeline = context.get_pipeline("sub_scalar", "scalar", (wsize, 1));

        let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (wsize * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false
        });

        let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (wsize * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let scalar_data = ArrayD::from_elem(IxDyn(&[1]), scalar);
        let scalar = GpuTensor::<T>::new_on_gpu(scalar_data, None);

        let bind_group = self.get_bind_group_zip(&scalar, &pipeline, &output_buffer);
        let result = self.encode_and_submit(&pipeline, &bind_group, &output_buffer, &staging_buffer, &self.shape()).wait();
        let output = result.map(|&e| T::from(e).unwrap());
        
        GpuTensor::new_on_gpu(output, None)
    }

    fn zeros(shape: &[usize]) -> Self {
        GpuTensor::new_on_gpu(ArrayD::zeros(shape), None)
    }

    fn zeros_like(tensor: &Self) -> Self {
        GpuTensor::new_on_gpu(ArrayD::zeros(tensor.data.raw_dim()), None)
    }
    fn ones(shape: &[usize]) -> Self {
        GpuTensor::new_on_gpu(ArrayD::ones(shape), None)
    }

    fn max_axis(&self, axis: usize) -> Self {
        let result = self.data.map_axis(Axis(axis), |x| {
            let mut max = T::from_f32(std::f32::NEG_INFINITY).unwrap();
            let mut max_index = 0.0;
            for (i, &val) in x.iter().enumerate() {
                if val > max {
                    max = val;
                    max_index = i as f32;
                }
            }
            T::from(max_index).unwrap()
        });

        GpuTensor::new_on_gpu(result, None)
    }

    fn mean_axis(&self, axis: usize) -> Self {
        GpuTensor::new_on_gpu(self.data.mean_axis(Axis(axis)).unwrap(), None)
    }

    fn sum_axis(&self, axis: usize) -> Self {
        GpuTensor::new_on_gpu(self.data.sum_axis(Axis(axis)), None)
    }

    fn add_axis(&self, axis: usize, values: &Array1<T>) -> Self {
        let binding = values.clone().insert_axis(Axis(axis));
        let values_broadcasted = binding.broadcast(self.data.raw_dim()).unwrap();
        let result = &self.data + &values_broadcasted;

        GpuTensor::new_on_gpu(result, None)
    }

    fn sub_axis(&self, axis: usize, values: &Array1<T>) -> Self {
        let binding = values.clone().insert_axis(Axis(axis));
        let values_broadcasted = binding.broadcast(self.data.raw_dim()).unwrap();
        let result = &self.data - &values_broadcasted;

        GpuTensor::new_on_gpu(result, None)
    }

    fn norm(&self) -> T {
        self.data.iter().map(|x| x.powi(2)).sum::<T>().sqrt()
    }

    fn variance(&self, axis: usize) -> Self {
        GpuTensor::new_on_gpu(self.data.var_axis(ndarray::Axis(axis), T::from(0.0).unwrap()), None)
    }

    fn sqrt(&self) -> Self {
        GpuTensor::new_on_gpu(self.data.map(|x| x.sqrt()), None)
    }

    fn pow(&self, exp: T) -> Self {
        GpuTensor::new_on_gpu(self.data.map(|x| x.powf(exp)), None)
    }

    fn glorot_uniform(shape: &[usize]) -> Self {
        let fan_in = shape[0];
        let fan_out = shape[1];
        let limit = (T::from(6.0).unwrap() / T::from(fan_in + fan_out).unwrap()).sqrt();
        let uniform = Uniform::<T>::new(-limit, limit);
        let mut rng = rand::thread_rng();
        let data: Vec<T> = uniform.sample_iter(&mut rng).take(fan_in * fan_out).collect();
        let data = Array::from_shape_vec((fan_in, fan_out), data).unwrap().into_dyn();

        GpuTensor::new_on_gpu(data, None)
    }

    fn he_uniform(shape: &[usize]) -> Self {
        let fan_in = shape[0];
        let limit = (T::from(2.0).unwrap() / T::from(fan_in).unwrap()).sqrt();
        let uniform = Uniform::new(-limit, limit);
        let mut rng = rand::thread_rng();
        let data: Vec<T> = uniform.sample_iter(&mut rng).take(shape.iter().product()).collect();

        GpuTensor::new_on_gpu(Array::from_shape_vec(shape, data).unwrap().into_dyn(), None)
    }

    fn random(shape: &[usize]) -> Self {
        let mut rng = rand::thread_rng();
        let data = Array::<T, IxDyn>::from_shape_fn(IxDyn(&shape), |_| {
            let a = T::from(-1.0).unwrap();
            let b = T::from(1.0).unwrap();
            rng.gen_range(a..b)
        });

        GpuTensor::new_on_gpu(data, None)
    }

    fn slice(&self, axis: usize, start: usize, end: usize) -> Self {
        GpuTensor::new_on_gpu(self.data.slice_axis(Axis(axis), (start..end).into()).to_owned(), None)
    }

    fn slice_mut(&mut self, axis: usize, start: usize, end: usize) -> ArrayViewMutD<T> {
        self.data.slice_axis_mut(Axis(axis), (start..end).into())
    }

    fn index_axis(&self, axis: usize, index: usize) -> ArrayViewD<T> {
        self.data.index_axis(Axis(axis), index)
    }

    fn index_axis_move(&self, axis: usize, index: usize) -> Self {
        GpuTensor::new_on_gpu(self.data.clone().index_axis_move(Axis(axis), index), None)
    }

    fn iter(&self) -> ndarray::iter::Iter<'_, T, IxDyn> {
        self.data.iter()
    }

    fn iter_mut(&mut self) -> ndarray::iter::IterMut<'_, T, IxDyn> {
        self.data.iter_mut()
    }

    fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }

    fn size(&self) -> usize {
        self.data.len()
    }

    fn raw_dim(&self) -> IxDyn {
        self.data.raw_dim()
    }

    fn onehot(target: &Self, num_classes: usize) -> Self {
        let mut one_hot = GpuTensor::zeros(&[target.shape()[0], target.shape()[1], num_classes]);
        for i in 0..target.shape()[0] {
            for j in 0..target.shape()[1] {
                let target_index = target.data[[i, j]].to_usize().unwrap();
                one_hot.data[[i, j, target_index]] = T::from(1.0).unwrap();
            }
        }
        one_hot
    }
}

impl<'a, T> GpuTensor<'a, T> 
where T: bytemuck::Pod + 'static
{
    pub fn new_on_gpu(data: ArrayD<T>, label: Option<&str>) -> Self {
        
        let context = get_gpu_context();
        let buffer_label = label.unwrap_or("GPU Tensor");

        let binding = data.as_standard_layout();
        let slice = binding.as_slice().unwrap();

        let buffer = context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(buffer_label),
            contents: bytemuck::cast_slice(slice),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
        });

        Self { data, buffer, context }
    }

    pub fn get_bind_group_zip(&self, other: &Self, pipeline: &wgpu::ComputePipeline, output_buffer: &wgpu::Buffer) -> wgpu::BindGroup {
        
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(self.buffer.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(other.buffer.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(output_buffer.as_entire_buffer_binding()),
                },
            ],
        });
        bind_group
    }

    pub fn get_bind_group_matmul(&self, batch: &Self, pipeline: &wgpu::ComputePipeline, output_buffer: &wgpu::Buffer) -> wgpu::BindGroup {
        let bshape = batch.data.shape();
        
        let n = bshape[0]; // number of rows
        let m = bshape[1]; // number of columns
        let b = 1; // batch = 1
    
        // Create a buffer for the Uniforms
        let uniforms = Uniforms {
            n: n as u32,
            m: m as u32,
            b: b as u32,
        };
    
        let uniforms_buffer = self.context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniforms Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
    
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(uniforms_buffer.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(batch.buffer.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(self.buffer.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(output_buffer.as_entire_buffer_binding()),
                },
            ],
        });
    
        bind_group
    }

    pub fn get_bind_group_batch(&self, batch: &Self, pipeline: &wgpu::ComputePipeline, output_buffer: &wgpu::Buffer) -> wgpu::BindGroup {
        let bshape = batch.data.shape();
        
        let n = bshape[1]; // number of rows
        let m = bshape[2]; // number of columns
        let b = bshape[0]; // number of matrices in batch
    
        // Create a buffer for the Uniforms
        let uniforms = Uniforms {
            n: n as u32,
            m: m as u32,
            b: b as u32,
        };
    
        let uniforms_buffer = self.context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniforms Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
    
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(uniforms_buffer.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(batch.buffer.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(self.buffer.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(output_buffer.as_entire_buffer_binding()),
                },
            ],
        });
    
        bind_group
    }

    pub async fn encode_and_submit(&self, compute_pipeline: &wgpu::ComputePipeline, bind_group: &wgpu::BindGroup, output_buffer: &wgpu::Buffer, staging_buffer: &wgpu::Buffer, target_shape: &[usize]) -> ArrayD<f32> {
        let mut encoder = self.context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        
        let workgroup_count_x: u32;
        let workgroup_count_y: u32;

        if target_shape.len() == 2 {
            workgroup_count_x = (target_shape[0] / 32 + 1) as u32;
            workgroup_count_y = (target_shape[1] / 32 + 1) as u32;
        } else if target_shape.len() == 1 {
            workgroup_count_x = (target_shape[0] / 32 + 1) as u32;
            workgroup_count_y = 1 as u32;
        } else {
            workgroup_count_x = 16 as u32;
            workgroup_count_y = 16 as u32;
        }
    
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            compute_pass.set_pipeline(compute_pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
        }
    
        encoder.copy_buffer_to_buffer(output_buffer, 0, staging_buffer, 0, (self.data.len() * std::mem::size_of::<f32>()) as u64);
        
        let index = self.context.queue.submit(Some(encoder.finish()));
    
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
    
        self.context.device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(index));
        receiver.receive().await.unwrap().unwrap();
    
        let data = buffer_slice.get_mapped_range();
        let data_f32 = bytemuck::cast_slice::<u8, f32>(&data);
        let result = ArrayD::<f32>::from_shape_vec(IxDyn(&target_shape), data_f32.to_vec()).unwrap();
        drop(data);
        staging_buffer.unmap();
    
        result
    }

    pub async fn encode_and_submit_batch(&self, compute_pipeline: &wgpu::ComputePipeline, bind_group: &wgpu::BindGroup, output_buffer: &wgpu::Buffer, staging_buffer: &wgpu::Buffer, target_shape: &[usize], wsize: usize) -> ArrayD<f32> {
        let mut encoder = self.context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let workgroup_count_x = ((target_shape[2] + 31) / 32) as u32;
        let workgroup_count_y = ((target_shape[1] + 31) / 32) as u32;
        let workgroup_count_z = target_shape[0] as u32;
    
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            compute_pass.set_pipeline(compute_pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, workgroup_count_z);
        }
    
        encoder.copy_buffer_to_buffer(output_buffer, 0, staging_buffer, 0, (wsize * std::mem::size_of::<T>()) as u64);
        
        self.context.queue.submit(Some(encoder.finish()));
    
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
    
        self.context.device.poll(wgpu::MaintainBase::Wait);
        receiver.receive().await.unwrap().unwrap();
    
        let data = buffer_slice.get_mapped_range();
        let data_f32 = bytemuck::cast_slice::<u8, f32>(&data);
        let result = ArrayD::<f32>::from_shape_vec(IxDyn(&target_shape), data_f32.to_vec()).unwrap();
        drop(data);
        staging_buffer.unmap();
    
        result
    }

    // pub fn matmul_optimized(batch: Self, multiplier: Self) -> Array<f32, IxDyn> {
    //     let context = get_gpu_context();
    //     let wshape = batch.data.shape();
    //     let wsize = wshape.iter().fold(1, |acc, x| acc * x);
    //     let pipeline: Arc<wgpu::ComputePipeline> = context.get_pipeline("batch_mul", "matmul", (32, 32));

    //     let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
    //         label: None,
    //         size: (wsize * std::mem::size_of::<T>()) as u64,
    //         usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    //         mapped_at_creation: false
    //     });

    //     let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
    //         label: None,
    //         size: (wsize * std::mem::size_of::<T>()) as u64,
    //         usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    //         mapped_at_creation: false
    //     });

    //     let bind_group = multiplier.get_bind_group_batch(&batch, &pipeline, &output_buffer);

    //     let result = multiplier.encode_and_submit_batch(&pipeline, &bind_group, &output_buffer, &staging_buffer, &wshape, wsize).wait();
        
    //     result
    // }
}

impl<'a, T> Clone for GpuTensor<'a, T>
where
    T: num_traits::Float + bytemuck::Pod
{
    fn clone(&self) -> Self {
        GpuTensor::new_on_gpu(self.data.clone(), None)
    }
}

impl<'a, T> Serialize for GpuTensor<'a, T>
where
    T: num_traits::Float + Serialize
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut s = serializer.serialize_struct("GpuTensor", 1)?;
        s.serialize_field("data", &self.data)?;
        s.end()
    }
}

impl<'a, T> Deserialize<'a> for GpuTensor<'a, T>
where
    T: num_traits::Float + Deserialize<'a> + bytemuck::Pod
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        let data = <ArrayD<T>>::deserialize(deserializer)?;
        Ok(GpuTensor::new_on_gpu(data, None))
    }
}

impl<'a, T> AddAssign for GpuTensor<'a, T>
where
    T: num_traits::Float + AddAssign
{
    fn add_assign(&mut self, other: Self) {
        self.data += &other.data;
    }
}

impl<'a, T> SubAssign for GpuTensor<'a, T>
where
    T: num_traits::Float + SubAssign
{
    fn sub_assign(&mut self, other: Self) {
        self.data -= &other.data;
    }
}

impl<'a, T> Add for GpuTensor<'a, T>
where
    T: num_traits::Float + Add + bytemuck::Pod
{
    type Output = Self;
    fn add(self, other: Self) -> Self {
        GpuTensor::new_on_gpu(self.data + other.data, None)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    n: u32,
    m: u32,
    b: u32,
}

pub trait Block {
    fn wait(self) -> <Self as futures::Future>::Output
        where Self: Sized, Self: futures::Future
    {
        futures::executor::block_on(self)
    }
}

impl<F,T> Block for F
    where F: futures::Future<Output = T>
{}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[tokio::test]
    async fn test_gpu_tensor_random() {
        init_gpu_context().await;
        let mut rng = rand::thread_rng();
        let data = ArrayD::<f32>::from_shape_fn(IxDyn(&[2, 3]), |_| {
            rng.gen_range(-1.0..1.0)
        });

        let tensor = GpuTensor::<f32>::new_on_gpu(data.clone(), None);

        assert_eq!(tensor.data, data);
    }

    #[tokio::test]
    async fn test_multiply_tensors() {
        init_gpu_context().await;
        let mut rng = rand::thread_rng();
        let data1 = ArrayD::<f32>::from_shape_fn(IxDyn(&[2, 3]), |_| {
            rng.gen_range(-1.0..1.0)
        });
        let data2 = ArrayD::<f32>::from_shape_fn(IxDyn(&[2, 3]), |_| {
            rng.gen_range(-1.0..1.0)
        });

        let context = get_gpu_context();
        let tensor = GpuTensor::<f32>::new_on_gpu(data1.clone(), None);
        let tensor2 = GpuTensor::<f32>::new_on_gpu(data2.clone(), None);

        let pipeline = context.get_pipeline("mul", "array", (6, 1));

        let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (2 * 3 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false
        });

        let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (2 * 3 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let bind_group = tensor.get_bind_group_zip(&tensor2, &pipeline, &output_buffer);
        let result = tensor.encode_and_submit(&pipeline, &bind_group, &output_buffer, &staging_buffer, data1.shape()).await;
        
        assert_eq!(result, data1 * data2);
    }

    #[tokio::test]
    async fn test_add_tensors() {
        init_gpu_context().await;
        let mut rng = rand::thread_rng();
        let data1 = ArrayD::<f32>::from_shape_fn(IxDyn(&[2, 3]), |_| {
            rng.gen_range(-1.0..1.0)
        });
        let data2 = ArrayD::<f32>::from_shape_fn(IxDyn(&[2, 3]), |_| {
            rng.gen_range(-1.0..1.0)
        });

        let context = get_gpu_context();
        let tensor = GpuTensor::<f32>::new_on_gpu(data1.clone(), None);
        let tensor2 = GpuTensor::<f32>::new_on_gpu(data2.clone(), None);

        let pipeline = context.get_pipeline("add", "array", (6, 1));

        let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (2 * 3 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false
        });

        let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (2 * 3 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let bind_group = tensor.get_bind_group_zip(&tensor2, &pipeline, &output_buffer);
        let result = tensor.encode_and_submit(&pipeline, &bind_group, &output_buffer, &staging_buffer, data1.shape()).await;
        
        assert_eq!(result, data1 + data2);
    }

    #[tokio::test]
    async fn test_multiply_with_scalar() {
        init_gpu_context().await;
        let mut rng = rand::thread_rng();
        let data = ArrayD::<f32>::from_shape_fn(IxDyn(&[2, 3]), |_| {
            rng.gen_range(-1.0..1.0)
        });
        let context = get_gpu_context();
        let tensor = GpuTensor::<f32>::new_on_gpu(data.clone(), None);
        let scalar = GpuTensor::<f32>::new_on_gpu(ArrayD::from_elem(IxDyn(&[1]), 2.0), None);

        let pipeline = context.get_pipeline("mul_scalar", "scalar", (6, 1));

        let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (2 * 3 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false
        });

        let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (2 * 3 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let bind_group = tensor.get_bind_group_zip(&scalar, &pipeline, &output_buffer);
        let result = tensor.encode_and_submit(&pipeline, &bind_group, &output_buffer, &staging_buffer, data.shape()).await;
        
        assert_eq!(result, data * 2.0);
    }

    #[tokio::test]
    async fn test_correct_results() {
        init_gpu_context().await;
        let mut rng = rand::thread_rng();
        let data1 = ArrayD::<f32>::from_shape_fn(IxDyn(&[2, 3]), |_| {
            rng.gen_range(-1.0..1.0)
        });
        let data2 = ArrayD::<f32>::from_shape_fn(IxDyn(&[2, 3]), |_| {
            rng.gen_range(-1.0..1.0)
        });

        let context = get_gpu_context();
        let tensor = GpuTensor::<f32>::new_on_gpu(data1.clone(), None);
        let tensor2 = GpuTensor::<f32>::new_on_gpu(data2.clone(), None);

        let pipeline = context.get_pipeline("mul", "array", (6, 1));

        let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (2 * 3 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false
        });

        let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (2 * 3 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let bind_group = tensor.get_bind_group_zip(&tensor2, &pipeline, &output_buffer);
        let result = tensor.encode_and_submit(&pipeline, &bind_group, &output_buffer, &staging_buffer, data1.shape()).await;
        
        assert_eq!(result, data1 * data2);
    }
}