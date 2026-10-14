// Shared layout: binding 0 = arena (activations, caches, small constants),
// binding 1 = weights (gemm / conv weights only). Strides are signed (flip = negative stride).
layout(std430, binding = 0) buffer A { float a[]; };
layout(std430, binding = 0) buffer A4 { vec4 a4[]; };
layout(std430, binding = 1) readonly buffer W { float w[]; };
layout(std430, binding = 1) readonly buffer W4 { vec4 w4[]; };
layout(push_constant) uniform P { uint u[32]; } p;
uint tid() { return gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * 65535u * gl_WorkGroupSize.x; }
float sigm(float x) { return 1.0 / (1.0 + exp(-x)); }
float epi(float v, uint op) {          // activation after a matmul / conv
    if (op == 1u) return v * clamp(v / 6.0 + 0.5, 0.0, 1.0);   // hardswish
    if (op == 2u) return v * sigm(v);                          // silu
    if (op == 3u) return max(v, 0.0);                          // relu
    if (op == 4u) return sigm(v);                              // sigmoid
    return v;
}
vec4 epi4(vec4 v, uint op) {
    if (op == 1u) return v * clamp(v / 6.0 + 0.5, 0.0, 1.0);
    if (op == 2u) return v / (1.0 + exp(-v));
    if (op == 3u) return max(v, vec4(0.0));
    if (op == 4u) return 1.0 / (1.0 + exp(-v));
    return v;
}
