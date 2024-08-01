@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> scalar: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute
@workgroup_size(256)
fn mul_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x;
    output[gidx] = input[gidx] * scalar[0];
}

@compute
@workgroup_size(256)
fn div_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x;
    output[gidx] = input[gidx] / scalar[0];
}

@compute
@workgroup_size(256)
fn add_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x;
    output[gidx] = input[gidx] + scalar[0];
}

@compute
@workgroup_size(256)
fn sub_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x;
    output[gidx] = input[gidx] - scalar[0];
}