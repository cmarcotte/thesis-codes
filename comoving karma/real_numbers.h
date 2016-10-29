// Maps real_t :> {float, double} based on precision in *.m file
// allows for single kernel file for multiple precisions
#if CONFIG_USE_DOUBLE

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#if defined(FP_FAST_FMA)
#pragma OPENCL FP_CONTRACT ON
#endif

// double
typedef double real;
typedef double2 real2;
typedef double3 real3;
typedef double4 real4;
typedef double8 real8;
typedef double16 real16;
#define FFT_PI          3.14159265358979323846
#define FFT_SQRT_1_2    0.70710678118654752440

#else

#if defined(FP_FAST_FMAF)
#pragma OPENCL FP_CONTRACT ON
#endif

// float
typedef float real;
typedef float2 real2;
typedef float3 real3;
typedef float4 real4;
typedef float8 real8;
typedef float16 real16;
#define FFT_PI       3.14159265359f
#define FFT_SQRT_1_2 0.707106781187f

#endif
