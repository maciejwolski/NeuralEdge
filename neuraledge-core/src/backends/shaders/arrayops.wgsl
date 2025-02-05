@group(0) @binding(0)
var<storage, read> input_0: array<f32>;

@group(0) @binding(1)
var<storage, read> input_1: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output_0: array<f32>;

fn div(a: f32, b: f32) -> f32 {
    let eps = 1e-8;
    return a / select(b, sign(b) * eps, abs(b) < eps);
}

fn safe_sqrt(x: f32) -> f32 {
    return sqrt(max(x, 0.0));
}

@compute
@workgroup_size(256)
fn mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x;
    output_0[gidx] = input_0[gidx] * input_1[gidx];
}

@compute
@workgroup_size(256)
fn divide(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gidx = global_id.x;
    output_0[gidx] = div(input_0[gidx], input_1[gidx]);
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