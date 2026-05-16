#ifndef MINIGPU_KERNEL_H
#define MINIGPU_KERNEL_H

#if defined(__INTELLISENSE__) || defined(__clangd__)
#define __global__
#define __device__
#define __host__
#define __shared__
#define __forceinline__ inline
#endif

#ifndef __global__
#define __global__ __attribute__((global))
#endif

#ifndef __device__
#define __device__ __attribute__((device))
#endif

#ifndef __host__
#define __host__ __attribute__((host))
#endif

#ifndef __shared__
#define __shared__ __attribute__((shared))
#endif

#ifndef __forceinline__
#define __forceinline__ __attribute__((always_inline)) inline
#endif

#define MINIGPU_INLINE static __device__ __forceinline__
#define MINIGPU_SHARED extern "C" __device__ __attribute__((noinline)) __attribute__((used)) __attribute__((annotate("minigpu_shared")))
#define MINIGPU_REGISTER_KERNEL(name, op, dtype, entry)
#define MINIGPU_REGISTER_TYPED_KERNELS(op, dtypes, entry)

struct dim3 {
  unsigned int x, y, z;
};

typedef signed char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned short half;
typedef unsigned short __half;
typedef unsigned char fp8;
typedef unsigned char fp8_e4m3;
typedef unsigned char fp8_e4m3fn;
typedef unsigned char __nv_fp8_e4m3;

extern __device__ const dim3 threadIdx;
extern __device__ const dim3 blockIdx;
extern __device__ const dim3 blockDim;
extern __device__ const dim3 gridDim;
extern __device__ void __syncthreads();
extern __device__ uint32_t minigpu_as_u32(float value);
extern __device__ float minigpu_as_f32(uint32_t value);
extern __device__ uint32_t __float_as_uint(float value);
extern __device__ float __uint_as_float(uint32_t value);

MINIGPU_SHARED uint32_t minigpu_mask_u32(int pred) {
  return 0u - (uint32_t)pred;
}

MINIGPU_SHARED uint32_t minigpu_select_u32(uint32_t mask, uint32_t if_set, uint32_t if_clear) {
  return (mask & if_set) | ((~mask) & if_clear);
}

MINIGPU_SHARED float minigpu_select_f32(uint32_t mask, float if_set, float if_clear) {
  return __uint_as_float(minigpu_select_u32(mask, __float_as_uint(if_set), __float_as_uint(if_clear)));
}

MINIGPU_SHARED int32_t minigpu_float_order_key(float x) {
  uint32_t bits = __float_as_uint(x);
  if (bits & 0x80000000u) {
    return (int32_t)((~bits) | 0x80000000u);
  }
  return (int32_t)bits;
}

MINIGPU_SHARED int minigpu_float_lt(float a, float b) {
  return minigpu_float_order_key(a) < minigpu_float_order_key(b);
}

MINIGPU_SHARED int minigpu_float_gt(float a, float b) {
  return minigpu_float_order_key(a) > minigpu_float_order_key(b);
}

MINIGPU_SHARED float fabsf(float x) {
  return __uint_as_float(__float_as_uint(x) & 0x7fffffffu);
}

MINIGPU_SHARED float fabs(float x) {
  return fabsf(x);
}

MINIGPU_SHARED float fminf(float a, float b) {
  float y = a;
  if (minigpu_float_lt(b, a)) {
    y = b;
  }
  return y;
}

MINIGPU_SHARED float fmaxf(float a, float b) {
  float y = a;
  if (minigpu_float_gt(b, a)) {
    y = b;
  }
  return y;
}

MINIGPU_SHARED float fmin(float a, float b) {
  return fminf(a, b);
}

MINIGPU_SHARED float fmax(float a, float b) {
  return fmaxf(a, b);
}

MINIGPU_SHARED float copysignf(float mag, float sign) {
  uint32_t mag_bits = __float_as_uint(mag) & 0x7fffffffu;
  uint32_t sign_bits = __float_as_uint(sign) & 0x80000000u;
  return __uint_as_float(mag_bits | sign_bits);
}

MINIGPU_SHARED float copysign(float mag, float sign) {
  return copysignf(mag, sign);
}

MINIGPU_SHARED float clampf(float x, float lo, float hi) {
  float y = x;
  if (minigpu_float_lt(y, lo)) {
    y = lo;
  }
  if (minigpu_float_gt(y, hi)) {
    y = hi;
  }
  return y;
}

MINIGPU_SHARED int minigpu_round_i32(float x) {
  return (int)(x + copysignf(0.5f, x));
}

MINIGPU_SHARED float roundf(float x) {
  return (float)minigpu_round_i32(x);
}

MINIGPU_SHARED float round(float x) {
  return roundf(x);
}

MINIGPU_SHARED float truncf(float x) {
  return (float)((int)x);
}

MINIGPU_SHARED float trunc(float x) {
  return truncf(x);
}

MINIGPU_SHARED float floorf(float x) {
  int i = (int)x;
  float y = (float)i;
  if (minigpu_float_gt(y, x)) {
    i = i - 1;
    y = (float)i;
  }
  return y;
}

MINIGPU_SHARED float floor(float x) {
  return floorf(x);
}

MINIGPU_SHARED float ceilf(float x) {
  int i = (int)x;
  float y = (float)i;
  if (minigpu_float_lt(y, x)) {
    i = i + 1;
    y = (float)i;
  }
  return y;
}

MINIGPU_SHARED float ceil(float x) {
  return ceilf(x);
}

MINIGPU_SHARED float rcpf(float x) {
  uint32_t ix = __float_as_uint(x);
  uint32_t sign = ix & 0x80000000u;
  uint32_t mag = ix & 0x7fffffffu;
  uint32_t iy = sign | (0x7ef311c3u - mag);
  float y = __uint_as_float(iy);
  float two = 2.0f;

  y = y * (two - x * y);
  y = y * (two - x * y);
  y = y * (two - x * y);

  uint32_t zero_mask = minigpu_mask_u32(mag == 0u);
  uint32_t inf_mask = minigpu_mask_u32(mag == 0x7f800000u);
  uint32_t nan_mask = minigpu_mask_u32(mag > 0x7f800000u);
  uint32_t result = __float_as_uint(y);
  result = minigpu_select_u32(zero_mask, sign | 0x7f800000u, result);
  result = minigpu_select_u32(inf_mask, sign, result);
  result = minigpu_select_u32(nan_mask, ix, result);
  return __uint_as_float(result);
}

MINIGPU_SHARED float __frcp_rn(float x) {
  return rcpf(x);
}

MINIGPU_SHARED float rsqrtf(float x) {
  uint32_t ix = __float_as_uint(x);
  uint32_t seed = 0x5f3759dfu - (ix >> 1);
  float y = __uint_as_float(seed);
  float half = 0.5f;
  float three_halves = 1.5f;

  y = y * (three_halves - half * x * y * y);
  y = y * (three_halves - half * x * y * y);
  y = y * (three_halves - half * x * y * y);
  return y;
}

MINIGPU_SHARED float sqrtf(float x) {
  uint32_t ix = __float_as_uint(x);
  uint32_t mag = ix & 0x7fffffffu;
  uint32_t sign = ix & 0x80000000u;
  float y = x * rsqrtf(x);

  uint32_t neg_mask = minigpu_mask_u32((sign != 0u) & (mag != 0u));
  uint32_t zero_mask = minigpu_mask_u32(mag == 0u);
  uint32_t inf_mask = minigpu_mask_u32((sign == 0u) & (mag == 0x7f800000u));
  uint32_t nan_mask = minigpu_mask_u32(mag > 0x7f800000u);
  uint32_t result = __float_as_uint(y);
  result = minigpu_select_u32(neg_mask, 0x7fc00000u, result);
  result = minigpu_select_u32(zero_mask, ix, result);
  result = minigpu_select_u32(inf_mask, ix, result);
  result = minigpu_select_u32(nan_mask, ix, result);
  return __uint_as_float(result);
}

MINIGPU_SHARED float sqrt(float x) {
  return sqrtf(x);
}

MINIGPU_SHARED float log2f(float x) {
  uint32_t raw = __float_as_uint(x);
  uint32_t ix = raw & 0x7fffffffu;
  uint32_t sign = raw & 0x80000000u;
  if (ix == 0u) {
    return __uint_as_float(0xff800000u);
  }
  int subnormal_shift = 0;
  if (ix < 0x00800000u) {
    x = x * 8388608.0f;
    ix = __float_as_uint(x) & 0x7fffffffu;
    subnormal_shift = -23;
  }

  int k = (int)(ix >> 23) - 127 + subnormal_shift;
  float y = __uint_as_float((ix & 0x007fffffu) | 0x3f800000u);

  float z = (y - 1.0f) * rcpf(y + 1.0f);
  float z2 = z * z;
  float z3 = z * z2;
  float z5 = z3 * z2;
  float z7 = z5 * z2;
  float ln_y = 2.0f * (z + 0.333333333333f * z3 +
      0.2f * z5 + 0.142857142857f * z7);
  float result_f = (float)k + 1.4426950408889634f * ln_y;
  uint32_t neg_mask = minigpu_mask_u32((sign != 0u) & (ix != 0u));
  uint32_t inf_mask = minigpu_mask_u32(ix == 0x7f800000u);
  uint32_t nan_mask = minigpu_mask_u32(ix > 0x7f800000u);
  uint32_t result = __float_as_uint(result_f);
  result = minigpu_select_u32(neg_mask, 0x7fc00000u, result);
  result = minigpu_select_u32(inf_mask, 0x7f800000u, result);
  result = minigpu_select_u32(nan_mask, raw, result);
  return __uint_as_float(result);
}

MINIGPU_SHARED float log2(float x) {
  return log2f(x);
}

MINIGPU_SHARED float logf(float x) {
  return 0.6931471805599453f * log2f(x);
}

MINIGPU_SHARED float log(float x) {
  return logf(x);
}

MINIGPU_SHARED float log10f(float x) {
  return 0.3010299956639812f * log2f(x);
}

MINIGPU_SHARED float log10(float x) {
  return log10f(x);
}

MINIGPU_SHARED float expf(float x) {
  uint32_t ix = __float_as_uint(x);
  uint32_t mag = ix & 0x7fffffffu;
  uint32_t sign = ix & 0x80000000u;
  float y = 1.0f + x * 0.00048828125f;
  y = y * y;
  y = y * y;
  y = y * y;
  y = y * y;
  y = y * y;
  y = y * y;
  y = y * y;
  y = y * y;
  y = y * y;
  y = y * y;
  y = y * y;

  uint32_t overflow_mask = minigpu_mask_u32((sign == 0u) & (mag > 0x42b17217u));
  uint32_t underflow_mask = minigpu_mask_u32((sign != 0u) & (mag > 0x42aeac50u));
  uint32_t nan_mask = minigpu_mask_u32(mag > 0x7f800000u);
  uint32_t result = __float_as_uint(y);
  result = minigpu_select_u32(overflow_mask, 0x7f800000u, result);
  result = minigpu_select_u32(underflow_mask, 0u, result);
  result = minigpu_select_u32(nan_mask, ix, result);
  return __uint_as_float(result);
}

MINIGPU_SHARED float exp(float x) {
  return expf(x);
}

MINIGPU_SHARED float exp2f(float x) {
  return expf(x * 0.6931471805599453f);
}

MINIGPU_SHARED float exp2(float x) {
  return exp2f(x);
}

MINIGPU_SHARED float powf(float base, float exponent) {
  return expf(exponent * logf(base));
}

MINIGPU_SHARED float pow(float base, float exponent) {
  return powf(base, exponent);
}

MINIGPU_SHARED float minigpu_xor_sign_f32(float x, uint32_t sign_mask) {
  return __uint_as_float(__float_as_uint(x) ^ (sign_mask & 0x80000000u));
}

MINIGPU_SHARED float minigpu_sin_poly(float r) {
  float r2 = r * r;
  float r3 = r2 * r;
  float r5 = r3 * r2;
  float r7 = r5 * r2;
  return r - 0.166666666667f * r3 + 0.008333333333f * r5 - 0.000198412698f * r7;
}

MINIGPU_SHARED float minigpu_cos_poly(float r) {
  float r2 = r * r;
  float r4 = r2 * r2;
  float r6 = r4 * r2;
  return 1.0f - 0.5f * r2 + 0.041666666667f * r4 - 0.001388888889f * r6;
}

MINIGPU_SHARED int minigpu_nearest_quadrant(float x) {
  return minigpu_round_i32(x * 0.6366197723675813f);
}

MINIGPU_SHARED float sinf(float x) {
  int q = minigpu_nearest_quadrant(x);
  int quadrant = q & 3;
  float r = x - (float)q * 1.5707963267948966f;
  float s = minigpu_sin_poly(r);
  float c = minigpu_cos_poly(r);
  uint32_t swap_mask = 0u - (uint32_t)(quadrant & 1);
  uint32_t sign_mask = 0u - (uint32_t)((quadrant >> 1) & 1);
  return minigpu_xor_sign_f32(minigpu_select_f32(swap_mask, c, s), sign_mask);
}

MINIGPU_SHARED float sin(float x) {
  return sinf(x);
}

MINIGPU_SHARED float cosf(float x) {
  int q = minigpu_nearest_quadrant(x);
  int quadrant = q & 3;
  float r = x - (float)q * 1.5707963267948966f;
  float s = minigpu_sin_poly(r);
  float c = minigpu_cos_poly(r);
  uint32_t swap_mask = 0u - (uint32_t)(quadrant & 1);
  uint32_t sign_mask = 0u - (uint32_t)(((quadrant + 1) >> 1) & 1);
  return minigpu_xor_sign_f32(minigpu_select_f32(swap_mask, s, c), sign_mask);
}

MINIGPU_SHARED float cos(float x) {
  return cosf(x);
}

MINIGPU_SHARED float tanf(float x) {
  return sinf(x) * rcpf(cosf(x));
}

MINIGPU_SHARED float tan(float x) {
  return tanf(x);
}

MINIGPU_SHARED float sigmoidf(float x) {
  uint32_t ix = __float_as_uint(x);
  uint32_t mag = ix & 0x7fffffffu;
  uint32_t sign = ix & 0x80000000u;
  float y = rcpf(1.0f + expf(0.0f - x));

  uint32_t pos_sat_mask = minigpu_mask_u32((sign == 0u) & (mag > 0x42b17217u));
  uint32_t neg_sat_mask = minigpu_mask_u32((sign != 0u) & (mag > 0x42b17217u));
  uint32_t nan_mask = minigpu_mask_u32(mag > 0x7f800000u);
  uint32_t result = __float_as_uint(y);
  result = minigpu_select_u32(pos_sat_mask, 0x3f800000u, result);
  result = minigpu_select_u32(neg_sat_mask, 0u, result);
  result = minigpu_select_u32(nan_mask, ix, result);
  return __uint_as_float(result);
}

MINIGPU_SHARED float tanhf(float x) {
  uint32_t ix = __float_as_uint(x);
  uint32_t mag = ix & 0x7fffffffu;
  uint32_t sign = ix & 0x80000000u;
  float y = 2.0f * sigmoidf(2.0f * x) - 1.0f;

  uint32_t sat_mask = minigpu_mask_u32(mag > 0x41200000u);
  uint32_t nan_mask = minigpu_mask_u32(mag > 0x7f800000u);
  uint32_t sat = sign | 0x3f800000u;
  uint32_t result = __float_as_uint(y);
  result = minigpu_select_u32(sat_mask, sat, result);
  result = minigpu_select_u32(nan_mask, ix, result);
  return __uint_as_float(result);
}

MINIGPU_SHARED float tanh(float x) {
  return tanhf(x);
}

#endif
