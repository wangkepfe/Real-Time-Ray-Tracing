#pragma once

#include <cuda_runtime.h>

// fast 4-byte integer hash by Thomas Wang
static __forceinline__ __device__ unsigned int WangHash(unsigned int a)
{
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

#if 1
typedef unsigned int curandDirectionVectors32_t[32];

struct curandStateScrambledSobol32 {
	unsigned int i, x, c;
	unsigned int direction_vectors[32];
};

typedef struct curandStateScrambledSobol32 curandStateScrambledSobol32_t;

static __forceinline__ __device__ void skipahead(unsigned int n, curandStateScrambledSobol32_t* state)
{
	unsigned int i_gray;
	state->x = state->c;
	state->i += n;
	/* Convert state->i to gray code */
	i_gray = state->i ^ (state->i >> 1);
	for (unsigned int k = 0; k < 32; k++) {
		if (i_gray & (1 << k)) {
			state->x ^= state->direction_vectors[k];
		}
	}
	return;
}

static __forceinline__ __device__ void curand_init(
	curandDirectionVectors32_t direction_vectors,
	unsigned int scramble_c,
	unsigned int offset,
	curandStateScrambledSobol32_t* state)
{
	state->i = 0;
	state->c = scramble_c;
	for (int i = 0; i < 32; i++) {
		state->direction_vectors[i] = direction_vectors[i];
	}
	state->x = state->c;
	skipahead(offset, state);
}

static __forceinline__ __device__ int __curand_find_trailing_zero(unsigned int x)
{
#if __CUDA_ARCH__ > 0
	int y = __ffs(~x);
	if (y)
		return y - 1;
	return 31;
#else
	int i = 1;
	while (x & 1) {
		i++;
		x >>= 1;
	}
	i = i - 1;
	return i == 32 ? 31 : i;
#endif
}

static __forceinline__ __device__ unsigned int curand(curandStateScrambledSobol32_t* state)
{
	/* Moving from i to i+1 element in gray code is flipping one bit,
	   the trailing zero bit of i
	*/
	unsigned int res = state->x;
	state->x ^= state->direction_vectors[__curand_find_trailing_zero(state->i)];
	state->i++;
	return res;
}

#define CURAND_2POW32_INV (2.3283064e-10f)
#define CURAND_2POW32 (4294967296.f)

static __forceinline__ __device__ float _curand_uniform(unsigned int x)
{
	return x * CURAND_2POW32_INV + (CURAND_2POW32_INV / 2.0f);
}

static __forceinline__ __device__ float curand_uniform(curandStateScrambledSobol32_t* state)
{
	return _curand_uniform(curand(state));
}

static uint32_t g_curandDirectionVectors32[3][32] =
{
	{ 3993797307, 1495470478, 920144624, 358017595, 171071185, 116254830, 45328601, 31154363, 11566025, 5813295, 3782404, 1407565, 1026410, 467345, 232745, 70134, 34597, 27448, 14474, 6533, 3921, 1402, 610, 458, 180, 125, 43, 31, 15, 5, 2, 1,},
	{ 3273207782, 2277097385, 3789678956, 3176721683, 3467502328, 2405445610, 4002629160, 3055395988, 3284744347, 2268498521, 3775884211, 3187491585, 3463818064, 2413857409, 4000792134, 3065294742, 3273252110, 2277050485, 3789634128, 3176686286, 3467451816, 2405470958, 4002651125, 3055363038, 3284781741, 2268478730, 3775903842, 3187526782, 3463795925, 2413836036, 4000834430, 3065260146,},
	{ 2259211747, 4283144984, 1349079232, 2484189875, 3696399352, 1703988716, 2319619859, 4205494334, 1510067397, 2590977361, 3663140577, 1851474051, 2262333737, 4288735829, 1357306915, 2487462320, 3702111207, 1696883867, 2319326687, 4199170979, 1519000081, 2586853817, 3668240289, 1860812437, 2259166711, 4283170192, 1349123107, 2484170083, 3696411015, 1703967229, 2319605968, 4205489904,},
};

#endif