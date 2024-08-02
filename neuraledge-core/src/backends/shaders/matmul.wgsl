struct Uniforms {
    n: u32,
    m: u32,
    b: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> matrices: array<f32>;
@group(0) @binding(2) var<storage, read> multiplier: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const WORKGROUP_SIZE_X: u32 = 32u;
const WORKGROUP_SIZE_Y: u32 = 32u;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y)
fn batch_mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = uniforms.n;  // number of rows in matrix
    let m = uniforms.m;  // number of columns in matrix
    let b = uniforms.b;  // number of matrices (batch size)
    
    let batch_index = global_id.z;
    let row = global_id.y;
    let col = global_id.x;
    
    if (batch_index >= b || row >= n || col >= m) {
        return;
    }
    
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < m; k = k + 1u) {
        let matrix_val = matrices[batch_index * n * m + row * m + k];
        let multiplier_val = multiplier[k * m + col];
        sum += matrix_val * multiplier_val;
    }
    
    output[batch_index * n * m + row * m + col] = sum;
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y)
fn matmul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = uniforms.n;  // number of rows in matrix A
    let m = uniforms.m;  // number of columns in matrix B
    let k = uniforms.b;  // number of columns in matrix A / rows in matrix B

    let row = global_id.y;
    let col = global_id.x;

    if (row >= n || col >= m) {
        return;
    }

    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < k; i = i + 1u) {
        let matrix_val = matrices[row * k + i];
        let multiplier_val = multiplier[i * m + col];
        sum += matrix_val * multiplier_val;
    }
    
    output[row * m + col] = sum;
}