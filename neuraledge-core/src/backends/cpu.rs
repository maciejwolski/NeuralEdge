use super::Tensor;
use ndarray::{Axis, ArrayD, Array, Array1, ArrayView, ArrayViewD, ArrayViewMutD, Ix2, Ix3, IxDyn};
use ndarray::linalg::general_mat_mul;
use rand::distributions::uniform::SampleUniform;
use serde::{Serialize, Deserialize};
use num_traits::{Float, FromPrimitive, Zero};

use rand::Rng;
use rand::distributions::{Distribution, Uniform};

use std::clone::Clone;
use std::ops::{Add, Sub, AddAssign, SubAssign};

use rayon::prelude::IntoParallelIterator;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuTensor<T> {
    pub data: ArrayD<T>,
}

impl<T> Tensor<T> for CpuTensor<T> 
where T: 'static + Float + Copy + Clone + Zero + SampleUniform + FromPrimitive + std::iter::Sum
{

    fn new(data: ArrayD<T>) -> Self {
        Self { data }
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
        Self { data: self.data.clone().reversed_axes() }
    }

    fn transpose_axes(&self, axis_1: usize, axis_2: usize) -> Self {
        let mut data = self.data.clone();
        data.swap_axes(axis_1, axis_2);
        Self { data }
    }

    fn insert_axis(&mut self, axis: usize) {
        self.set_data(self.data.clone().insert_axis(Axis(axis)));
    }

    fn stack(tensors: &[&Self], axis: usize) -> Self {
        let stacked_data = tensors
            .iter()
            .map(|tensor| tensor.data.view())
            .collect::<Vec<ArrayViewD<T>>>();
        Self { data: ndarray::stack(ndarray::Axis(axis), stacked_data.as_slice()).unwrap() }
    }

    fn concatenate(tensors: &[&Self], axis: usize) -> Self {
        let concatenated_data = tensors
            .iter()
            .map(|tensor| tensor.data.view())
            .collect::<Vec<ArrayViewD<T>>>();
        Self { data: ndarray::concatenate(Axis(axis), &concatenated_data).unwrap() }
    }

    fn split(&self, axis: usize, num_parts: usize) -> Vec<Self> {
        let axis_size = self.data.shape()[axis];
        let step = (axis_size as f32 / num_parts as f32).ceil();
        
        let mut result = Vec::new();
        for i in 0..num_parts {
            let start = i * step as usize;
            let end = (i + 1) * step as usize;
            let split_data = self.data.slice_axis(Axis(axis), (start..end).into()).to_owned();
            result.push(Tensor::new(split_data));
        }

        result
    }

    fn matmul(&self, other: &Self) -> Self {
        match (self.data.ndim(), other.data.ndim()) {
            (2,2) => {
                let first: Array<T, Ix2> = self.data.clone().into_dimensionality::<Ix2>().unwrap();
                let second: Array<T, Ix2> = other.data.clone().into_dimensionality::<Ix2>().unwrap();

                let mut result_data = Array::<T, Ix2>::zeros((first.shape()[0], second.shape()[1]));
                general_mat_mul(T::one(), &first, &second, T::one(), &mut result_data);
                let result: ArrayD<T> = result_data.into_dimensionality::<IxDyn>().unwrap();
                CpuTensor::new(result)
            },
            (3,3) => {
                let first: Array<T, Ix3> = self.data.clone().into_dimensionality::<Ix3>().unwrap();
                let second: Array<T, Ix3> = other.data.clone().into_dimensionality::<Ix3>().unwrap();

                let mut result_data = Array::<T, Ix3>::zeros((first.shape()[0], first.shape()[1], second.shape()[2]));
                for i in 0..first.shape()[0] {
                    let mut result_slice = result_data.index_axis_mut(Axis(0), i);
                    let first_clone = first.clone();
                    let second_clone = second.clone();
                    general_mat_mul(
                        T::one(),
                        &first_clone.index_axis_move(Axis(0), i),
                        &second_clone.index_axis_move(Axis(0), i),
                        T::one(),
                        &mut result_slice,
                    );
                }
                let result: ArrayD<T> = result_data.into_dimensionality::<IxDyn>().unwrap();
                CpuTensor::new(result)
            },
            (2,3) => {
                let first: Array<T, Ix2> = self.data.clone().into_dimensionality::<Ix2>().unwrap();
                let second: Array<T, Ix3> = other.data.clone().into_dimensionality::<Ix3>().unwrap();

                let mut result_data = Array::<T, Ix3>::zeros((second.shape()[0], first.shape()[0], second.shape()[2]));
                for i in 0..second.shape()[0] {
                    let second_clone = second.clone();
                    let mut result_slice = result_data.index_axis_mut(Axis(0), i);
                    general_mat_mul(
                        T::one(), &first, &second_clone.index_axis_move(Axis(0), i), T::one(), &mut result_slice
                    );
                }
                let result: ArrayD<T> = result_data.into_dimensionality::<IxDyn>().unwrap();
                CpuTensor::new(result)
            },
            (3,2) => {
                let first: Array<T, Ix3> = self.data.clone().into_dimensionality::<Ix3>().unwrap();
                let second: Array<T, Ix2> = other.data.clone().into_dimensionality::<Ix2>().unwrap();

                let mut result_data = Array::<T, Ix3>::zeros((first.shape()[0], first.shape()[1], second.shape()[1]));
                for i in 0..first.shape()[0] {
                    let first_slice = first.index_axis(Axis(0), i);
                    general_mat_mul(
                        T::one(), &first_slice, &second, T::one(), &mut result_data.index_axis_mut(Axis(0), i)
                    );
                }
                let result: ArrayD<T> = result_data.into_dimensionality::<IxDyn>().unwrap();
                CpuTensor::new(result)
            }
            _ => {
                panic!("Unsupported dimensionality for matrix multiplication");
            }
        }
    }

    fn batch_mul(&self, other: &Self) -> Self {
        let mut result_data = Vec::new();

        for i in 0..self.data.shape()[0] {
            let first_slice = self.data.index_axis(Axis(0), i).into_dimensionality::<Ix2>().unwrap().to_owned();
            let second_slice = other.data.index_axis(Axis(0), i).into_dimensionality::<Ix2>().unwrap().to_owned();
            let multiple = first_slice * second_slice; 
            result_data.push(multiple);
        }

        Self { data: Array::from_shape_vec(self.data.shape(), result_data.into_iter().flatten().collect()).unwrap().into_dyn() }
    }

    fn mul(&self, other: &Self) -> Self {
        Self { data: self.data.clone() * &other.data }
    }

    fn div(&self, other: &Self) -> Self {
        Self { data: self.data.clone() / &other.data }
    }

    fn add(&self, other: &Self) -> Self {
        Self { data: self.data.clone() + &other.data }
    }

    fn sub(&self, other: &Self) -> Self {
        Self { data: self.data.clone() - &other.data }
    }

    fn mul_scalar(&self, scalar: T) -> Self {
        Self { data: self.data.map(|x| *x * scalar) }
    }

    fn div_scalar(&self, scalar: T) -> Self {
        Self { data: self.data.map(|x| *x / scalar) }
    }

    fn add_scalar(&self, scalar: T) -> Self {
        Self { data: self.data.map(|x| *x + scalar) }
    }

    fn sub_scalar(&self, scalar: T) -> Self {
        Self { data: self.data.map(|x| *x - scalar) }
    }

    fn zeros(shape: &[usize]) -> Self {
        Self { data: Array::zeros(shape).into_dyn() }
    }

    fn zeros_like(tensor: &Self) -> Self {
        Self { data: Array::zeros(tensor.data.raw_dim()).into_dyn() }
    }

    fn ones(shape: &[usize]) -> Self {
        Self { data: Array::ones(shape).into_dyn() }
    }

    fn max_axis(&self, axis: usize) -> Self {
        let result = self.data.map_axis(Axis(axis), |x| {
            let mut max = T::from_f32(std::f32::NEG_INFINITY).unwrap();
            let mut max_index = 0;
            for (i, &val) in x.iter().enumerate() {
                if val > max {
                    max = val;
                    max_index = i;
                }
            }
            T::from(max_index).unwrap()
        });
        Self { data: result }
    }

    fn mean_axis(&self, axis: usize) -> Self {
        Self { data: self.data.mean_axis(Axis(axis)).unwrap() }
    }

    fn sum_axis(&self, axis: usize) -> Self {
        Self { data: self.data.sum_axis(Axis(axis)) }
    }

    fn add_axis(&self, axis: usize, values: &Array1<T>) -> Self {
        let binding = values.clone().insert_axis(Axis(axis));
        let values_broadcasted = binding.broadcast(self.data.raw_dim()).unwrap();
        let result = &self.data + &values_broadcasted;
        Self { data: result }
    }

    fn sub_axis(&self, axis: usize, values: &Array1<T>) -> Self {
        let binding = values.clone().insert_axis(Axis(axis));
        let values_broadcasted = binding.broadcast(self.data.raw_dim()).unwrap();
        let result = &self.data - &values_broadcasted;
        Self { data: result }
    }

    fn norm(&self) -> T {
        self.data.iter().map(|x| x.powi(2)).sum::<T>().sqrt()
    }

    fn variance(&self, axis: usize) -> Self {
        Self { data: self.data.var_axis(ndarray::Axis(axis), T::from(0.0).unwrap()) }
    }

    fn sqrt(&self) -> Self {
        Self { data: self.data.map(|x| x.sqrt()) }
    }

    fn pow(&self, exp: T) -> Self {
        Self { data: self.data.map(|x| x.powf(exp)) }
    }

    fn glorot_uniform(shape: &[usize]) -> Self {
        let fan_in = shape[0];
        let fan_out = shape[1];
        let limit = (T::from(6.0).unwrap() / T::from(fan_in + fan_out).unwrap()).sqrt();
        let uniform = Uniform::<T>::new(-limit, limit);
        let mut rng = rand::thread_rng();
        let data: Vec<T> = uniform.sample_iter(&mut rng).take(fan_in * fan_out).collect();
        let data = Array::from_shape_vec((fan_in, fan_out), data).unwrap().into_dyn();
        Self { data }
    }

    fn he_uniform(shape: &[usize]) -> Self
    {
        let fan_in = shape[0];
        let limit = (T::from(2.0).unwrap() / T::from(fan_in).unwrap()).sqrt();
        let uniform = Uniform::new(-limit, limit);
        let mut rng = rand::thread_rng();
        let data: Vec<T> = uniform.sample_iter(&mut rng).take(shape.iter().product()).collect();
        Self{ data: Array::from_shape_vec(shape, data).unwrap().into_dyn() }
    }

    fn random(shape: &[usize]) -> Self {
        let mut rng = rand::thread_rng();
        let data = Array::<T, IxDyn>::from_shape_fn(IxDyn(&shape), |_| {
            let a = T::from(-1.0).unwrap();
            let b = T::from(1.0).unwrap();
            rng.gen_range(a..b)
        });
        Self { data }
    }

    fn slice(&self, axis: usize, start: usize, end: usize) -> Self {
        Self { data: self.data.slice_axis(Axis(axis), (start..end).into()).to_owned() }
    }

    fn slice_mut(&mut self, axis: usize, start: usize, end: usize) -> ArrayViewMutD<T> {
        self.data.slice_axis_mut(Axis(axis), (start..end).into())
    }

    fn index_axis(&self, axis: usize, index: usize) -> ArrayViewD<T> {
        self.data.index_axis(Axis(axis), index)
    }

    fn index_axis_move(&self, axis: usize, index: usize) -> Self {
        Self { data: self.data.clone().index_axis_move(Axis(axis), index) }
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
        let mut one_hot = CpuTensor::zeros(&[target.shape()[0], target.shape()[1], num_classes]);
        for i in 0..target.shape()[0] {
            for j in 0..target.shape()[1] {
                let target_index = target.data[[i, j]].to_usize().unwrap();
                one_hot.data[[i, j, target_index]] = T::from(1.0).unwrap();
            }
        }
        one_hot
    }
}

impl<T> AddAssign for CpuTensor<T> 
where T: Float + AddAssign + Add,
{
    fn add_assign(&mut self, other: Self) {
        if self.data.ndim() == 3 && other.data.ndim() == 2 {
            self.data.map_axis(Axis(0), |x| x.add(&other.data));
        } else {
            self.data += &other.data;
        }
    }
}

impl<T> SubAssign for CpuTensor<T>
where T: Float + SubAssign + Sub,
{
    fn sub_assign(&mut self, other: Self) {
        if self.data.ndim() == 3 && other.data.ndim() == 2 {
            self.data.map_axis(Axis(0), |x| x.sub(&other.data));
        } else {
            self.data -= &other.data;
        }
    }
}

impl<T> Add for CpuTensor<T> 
where T: Float + SubAssign + Add,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            data: self.data + other.data,
        }
    }
}

impl<'a, T> IntoParallelIterator for &'a CpuTensor<T>
where
    T: Sync,
{
    type Item = &'a T;
    type Iter = ndarray::parallel::Parallel<ArrayView<'a, T, IxDyn>>;

    fn into_par_iter(self) -> Self::Iter {
        self.data.view().into_par_iter()
    }
}