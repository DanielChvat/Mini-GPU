#ifndef MINIGPU_KERNEL_H
#define MINIGPU_KERNEL_H

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

#define MINIGPU_MATH_INLINE static __device__ __forceinline__

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

MINIGPU_MATH_INLINE float fabsf(float x) {
  return __uint_as_float(__float_as_uint(x) & 0x7fffffffu);
}

MINIGPU_MATH_INLINE float fabs(float x) {
  return fabsf(x);
}

MINIGPU_MATH_INLINE float fminf(float a, float b) {
  float y = a;
  if (b < a) {
    y = b;
  }
  return y;
}

MINIGPU_MATH_INLINE float fmaxf(float a, float b) {
  float y = a;
  if (b > a) {
    y = b;
  }
  return y;
}

MINIGPU_MATH_INLINE float fmin(float a, float b) {
  return fminf(a, b);
}

MINIGPU_MATH_INLINE float fmax(float a, float b) {
  return fmaxf(a, b);
}

MINIGPU_MATH_INLINE float copysignf(float mag, float sign) {
  uint32_t mag_bits = __float_as_uint(mag) & 0x7fffffffu;
  uint32_t sign_bits = __float_as_uint(sign) & 0x80000000u;
  return __uint_as_float(mag_bits | sign_bits);
}

MINIGPU_MATH_INLINE float copysign(float mag, float sign) {
  return copysignf(mag, sign);
}

MINIGPU_MATH_INLINE float clampf(float x, float lo, float hi) {
  float y = x;
  if (y < lo) {
    y = lo;
  }
  if (y > hi) {
    y = hi;
  }
  return y;
}

MINIGPU_MATH_INLINE int minigpu_round_i32(float x) {
  return (int)(x + copysignf(0.5f, x));
}

MINIGPU_MATH_INLINE float roundf(float x) {
  return (float)minigpu_round_i32(x);
}

MINIGPU_MATH_INLINE float round(float x) {
  return roundf(x);
}

MINIGPU_MATH_INLINE float truncf(float x) {
  return (float)((int)x);
}

MINIGPU_MATH_INLINE float trunc(float x) {
  return truncf(x);
}

MINIGPU_MATH_INLINE float floorf(float x) {
  int i = (int)x;
  float y = (float)i;
  if (y > x) {
    i = i - 1;
    y = (float)i;
  }
  return y;
}

MINIGPU_MATH_INLINE float floor(float x) {
  return floorf(x);
}

MINIGPU_MATH_INLINE float ceilf(float x) {
  int i = (int)x;
  float y = (float)i;
  if (y < x) {
    i = i + 1;
    y = (float)i;
  }
  return y;
}

MINIGPU_MATH_INLINE float ceil(float x) {
  return ceilf(x);
}

MINIGPU_MATH_INLINE float rcpf(float x) {
  uint32_t ix = __float_as_uint(x);
  uint32_t sign = ix & 0x80000000u;
  uint32_t mag = ix & 0x7fffffffu;
  uint32_t iy = sign | (0x7ef311c3u - mag);
  float y = __uint_as_float(iy);
  float two = 2.0f;

  y = y * (two - x * y);
  y = y * (two - x * y);
  y = y * (two - x * y);
  return y;
}

MINIGPU_MATH_INLINE float __frcp_rn(float x) {
  return rcpf(x);
}

MINIGPU_MATH_INLINE float rsqrtf(float x) {
  uint32_t ix = __float_as_uint(x);
  uint32_t seed = 0x5f3759dfu - (ix >> 1);
  float y = __uint_as_float(seed);
  float half = 0.5f;
  float three_halves = 1.5f;

  y = y * (three_halves - half * x * y * y);
  y = y * (three_halves - half * x * y * y);
  y = y * (three_halves - half * x * y * y);
  if (x <= 0.0f) {
    y = 0.0f;
  }
  return y;
}

MINIGPU_MATH_INLINE float sqrtf(float x) {
  return x * rsqrtf(x);
}

MINIGPU_MATH_INLINE float sqrt(float x) {
  return sqrtf(x);
}

MINIGPU_MATH_INLINE float log2f(float x) {
  if (x == 1.0f) {
    return 0.0f;
  }
  if (x == 2.0f) {
    return 1.0f;
  }
  if (x == 4.0f) {
    return 2.0f;
  }
  if (x == 0.5f) {
    return -1.0f;
  }

  float y = x;
  float k = 0.0f;
  if (y > 2.0f) {
    y = y * 0.5f;
    k = k + 1.0f;
  }
  if (y > 2.0f) {
    y = y * 0.5f;
    k = k + 1.0f;
  }
  if (y < 1.0f) {
    y = y * 2.0f;
    k = k - 1.0f;
  }

  float z = (y - 1.0f) * rcpf(y + 1.0f);
  float z2 = z * z;
  float z3 = z * z2;
  float z5 = z3 * z2;
  float z7 = z5 * z2;
  float ln_y = 2.0f * (z + 0.333333333333f * z3 +
      0.2f * z5 + 0.142857142857f * z7);
  return k + 1.4426950408889634f * ln_y;
}

MINIGPU_MATH_INLINE float log2(float x) {
  return log2f(x);
}

MINIGPU_MATH_INLINE float logf(float x) {
  return 0.6931471805599453f * log2f(x);
}

MINIGPU_MATH_INLINE float log(float x) {
  return logf(x);
}

MINIGPU_MATH_INLINE float log10f(float x) {
  return 0.3010299956639812f * log2f(x);
}

MINIGPU_MATH_INLINE float log10(float x) {
  return log10f(x);
}

MINIGPU_MATH_INLINE float expf(float x) {
  float scale = 1.0f;
  float r = x;
  if (r > 1.0f) {
    r = r - 0.6931471805599453f;
    scale = scale * 2.0f;
  }
  if (r > 1.0f) {
    r = r - 0.6931471805599453f;
    scale = scale * 2.0f;
  }
  if (r < -1.0f) {
    r = r + 0.6931471805599453f;
    scale = scale * 0.5f;
  }
  if (r < -1.0f) {
    r = r + 0.6931471805599453f;
    scale = scale * 0.5f;
  }

  float r2 = r * r;
  float r3 = r2 * r;
  float r4 = r2 * r2;
  float r5 = r4 * r;
  float r6 = r3 * r3;
  float exp_r = 1.0f + r + 0.5f * r2 + 0.166666666667f * r3 +
      0.041666666667f * r4 + 0.008333333333f * r5 + 0.001388888889f * r6;
  return scale * exp_r;
}

MINIGPU_MATH_INLINE float exp(float x) {
  return expf(x);
}

MINIGPU_MATH_INLINE float exp2f(float x) {
  return expf(x * 0.6931471805599453f);
}

MINIGPU_MATH_INLINE float exp2(float x) {
  return exp2f(x);
}

MINIGPU_MATH_INLINE float powf(float base, float exponent) {
  return expf(exponent * logf(base));
}

MINIGPU_MATH_INLINE float pow(float base, float exponent) {
  return powf(base, exponent);
}

MINIGPU_MATH_INLINE float sinf(float x) {
  float r = x;
  float r2 = r * r;
  float r3 = r2 * r;
  float r5 = r3 * r2;
  float r7 = r5 * r2;
  return r - 0.166666666667f * r3 + 0.008333333333f * r5 - 0.000198412698f * r7;
}

MINIGPU_MATH_INLINE float sin(float x) {
  return sinf(x);
}

MINIGPU_MATH_INLINE float cosf(float x) {
  float r = x;
  float r2 = r * r;
  float r4 = r2 * r2;
  float r6 = r4 * r2;
  return 1.0f - 0.5f * r2 + 0.041666666667f * r4 - 0.001388888889f * r6;
}

MINIGPU_MATH_INLINE float cos(float x) {
  return cosf(x);
}

MINIGPU_MATH_INLINE float tanf(float x) {
  return sinf(x) * rcpf(cosf(x));
}

MINIGPU_MATH_INLINE float tan(float x) {
  return tanf(x);
}

MINIGPU_MATH_INLINE float sigmoidf(float x) {
  return rcpf(1.0f + expf(0.0f - x));
}

MINIGPU_MATH_INLINE float tanhf(float x) {
  float e = expf((0.0f - 2.0f) * x);
  return 2.0f * rcpf(1.0f + e) - 1.0f;
}

MINIGPU_MATH_INLINE float tanh(float x) {
  return tanhf(x);
}

#endif
