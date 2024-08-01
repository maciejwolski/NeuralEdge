@group(0) @binding(0)
var<storage, read> input_0: array<f32>;

@group(0) @binding(1)
var<storage, read> input_1: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output_0: array<f32>;

@compute
@workgroup_size(256)
fn mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x;
    output_0[gidx] = input_0[gidx] * input_1[gidx];
}

@compute
@workgroup_size(256)
fn div(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x;
    output_0[gidx] = input_0[gidx] / input_1[gidx];
}

@compute
@workgroup_size(256)
fn add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x;
    output_0[gidx] = input_0[gidx] + input_1[gidx];
}

@compute
@workgroup_size(256)
fn sub(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x;
    output_0[gidx] = input_0[gidx] - input_1[gidx];
}