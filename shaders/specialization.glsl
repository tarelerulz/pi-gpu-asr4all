// Fixed dimensions/strides/operations are set once at pipeline creation.
// UINT_MAX keeps the generic push-constant path. Buffer addresses remain dynamic.
layout(constant_id = 0) const uint shape_0 = 0xffffffffu;
layout(constant_id = 1) const uint shape_1 = 0xffffffffu;
layout(constant_id = 2) const uint shape_2 = 0xffffffffu;
layout(constant_id = 3) const uint shape_3 = 0xffffffffu;
layout(constant_id = 4) const uint shape_4 = 0xffffffffu;
layout(constant_id = 5) const uint shape_5 = 0xffffffffu;
layout(constant_id = 6) const uint shape_6 = 0xffffffffu;
layout(constant_id = 8) const uint shape_8 = 0xffffffffu;
layout(constant_id = 10) const uint shape_10 = 0xffffffffu;
layout(constant_id = 11) const uint shape_11 = 0xffffffffu;
layout(constant_id = 12) const uint shape_12 = 0xffffffffu;
layout(constant_id = 13) const uint shape_13 = 0xffffffffu;
layout(constant_id = 14) const uint shape_14 = 0xffffffffu;
layout(constant_id = 15) const uint shape_15 = 0xffffffffu;
layout(constant_id = 19) const uint shape_19 = 0xffffffffu;
layout(constant_id = 20) const uint shape_20 = 0xffffffffu;
layout(constant_id = 21) const uint shape_21 = 0xffffffffu;
layout(constant_id = 22) const uint shape_22 = 0xffffffffu;
#define U(k) ((shape_##k == 0xffffffffu) ? p.u[k] : shape_##k)
