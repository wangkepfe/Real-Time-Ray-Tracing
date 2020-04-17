#pragma once

#include <math.h>
#include <cuda_runtime.h>

#define uint unsigned int

#define PI_OVER_4               0.7853981633974483096156608458198757210492f
#define PI_OVER_2               1.5707963267948966192313216916397514420985f
#define SQRT_OF_ONE_THIRD       0.5773502691896257645091487805019574556476f
#define PI                      3.1415926535897932384626422832795028841971f
#define TWO_PI                  6.2831853071795864769252867665590057683943f
#define Pi_over_180             0.01745329251f
#define INV_PI                  0.31830988618f
#define INV_TWO_PI              0.15915494309f

struct Int2;

struct Float2
{
	union {
		struct { float x, y; };
		float _v[2];
	};

	__host__ __device__ Float2()                   : x(0), y(0)     {}
	__host__ __device__ Float2(float _x, float _y) : x(_x), y(_y)   {}
	__host__ __device__ Float2(float _x)           : x(_x), y(_x)   {}

	inline __host__ __device__ Float2  operator+(const Float2& v) const { return Float2(x + v.x, y + v.y); }
	inline __host__ __device__ Float2  operator-(const Float2& v) const { return Float2(x - v.x, y - v.y); }
	inline __host__ __device__ Float2  operator*(const Float2& v) const { return Float2(x * v.x, y * v.y); }
	inline __host__ __device__ Float2  operator/(const Float2& v) const { return Float2(x / v.x, y / v.y); }

	inline __host__ __device__ Float2  operator+(float a) const         { return Float2(x + a, y + a); }
	inline __host__ __device__ Float2  operator-(float a) const         { return Float2(x - a, y - a); }
	inline __host__ __device__ Float2  operator*(float a) const         { return Float2(x * a, y * a); }
	inline __host__ __device__ Float2  operator/(float a) const         { return Float2(x / a, y / a); }

	inline __host__ __device__ Float2& operator+=(const Float2& v)      { x += v.x; y += v.y; return *this; }
	inline __host__ __device__ Float2& operator-=(const Float2& v)      { x -= v.x; y -= v.y; return *this; }
	inline __host__ __device__ Float2& operator*=(const Float2& v)      { x *= v.x; y *= v.y; return *this; }
	inline __host__ __device__ Float2& operator/=(const Float2& v)      { x /= v.x; y /= v.y; return *this; }

	inline __host__ __device__ Float2& operator+=(const float& a)       { x += a; y += a; return *this; }
	inline __host__ __device__ Float2& operator-=(const float& a)       { x -= a; y -= a; return *this; }
	inline __host__ __device__ Float2& operator*=(const float& a)       { x *= a; y *= a; return *this; }
	inline __host__ __device__ Float2& operator/=(const float& a)       { x /= a; y /= a; return *this; }

	inline __host__ __device__ Float2 operator-() const { return Float2(-x, -y); }

	inline __host__ __device__ bool operator!=(const Float2& v) const   { return x != v.x || y != v.y; }
	inline __host__ __device__ bool operator==(const Float2& v) const   { return x == v.x && y == v.y; }

	inline __host__ __device__ float& operator[](int i)                 { return _v[i]; }
	inline __host__ __device__ float  operator[](int i) const           { return _v[i]; }

};

inline __host__ __device__ Float2 operator+(float a, const Float2& v)   { return Float2(v.x + a, v.y + a); }
inline __host__ __device__ Float2 operator-(float a, const Float2& v)   { return Float2(a - v.x, a - v.y); }
inline __host__ __device__ Float2 operator*(float a, const Float2& v)   { return Float2(v.x * a, v.y * a); }
inline __host__ __device__ Float2 operator/(float a, const Float2& v)   { return Float2(a / v.x, a / v.y); }

struct Int2
{
	union {
		struct { int x, y; };
		int _v[2];
	};

	__host__ __device__ Int2()              : x{0}, y{0} {}
	__host__ __device__ Int2(int a)         : x{a}, y{a} {}
    __host__ __device__ Int2(int x, int y)  : x{x}, y{y} {}

    inline __host__ __device__ Int2 operator + (int a) const         { return Int2(x + a, y + a); }
    inline __host__ __device__ Int2 operator - (int a) const         { return Int2(x - a, y - a); }

	inline __host__ __device__ Int2 operator += (int a)              { x += a; y += a; return *this; }
    inline __host__ __device__ Int2 operator -= (int a)              { x -= a; y -= a; return *this; }

    inline __host__ __device__ Int2 operator + (const Int2& v) const { return Int2(x + v.x, y + v.y); }
    inline __host__ __device__ Int2 operator - (const Int2& v) const { return Int2(x - v.x, y - v.y); }

	inline __host__ __device__ Int2 operator += (const Int2& v)      { x += v.x; y += v.y; return *this; }
    inline __host__ __device__ Int2 operator -= (const Int2& v)      { x -= v.x; y -= v.y; return *this; }

	inline __host__ __device__ Int2 operator - () const              { return Int2(-x, -y); }

    inline __host__ __device__ bool operator == (const Int2& v)      { return x == v.x && y == v.y; }
	inline __host__ __device__ bool operator != (const Int2& v)      { return x != v.x || y != v.y; }

	inline __host__ __device__ int& operator[] (int i)               { return _v[i]; }
	inline __host__ __device__ int  operator[] (int i) const         { return _v[i]; }
};

struct UInt2
{
	union {
		struct { unsigned int x, y; };
		unsigned int _v[2];
	};

	__host__ __device__ UInt2()              : x{0}, y{0} {}
	__host__ __device__ UInt2(unsigned int a)         : x{a}, y{a} {}
    __host__ __device__ UInt2(unsigned int x, unsigned int y)  : x{x}, y{y} {}

    inline __host__ __device__ UInt2 operator + (unsigned int a) const         { return UInt2(x + a, y + a); }
    inline __host__ __device__ UInt2 operator - (unsigned int a) const         { return UInt2(x - a, y - a); }

	inline __host__ __device__ UInt2 operator += (unsigned int a)              { x += a; y += a; return *this; }
    inline __host__ __device__ UInt2 operator -= (unsigned int a)              { x -= a; y -= a; return *this; }

    inline __host__ __device__ UInt2 operator + (const UInt2& v) const { return UInt2(x + v.x, y + v.y); }
    inline __host__ __device__ UInt2 operator - (const UInt2& v) const { return UInt2(x - v.x, y - v.y); }

	inline __host__ __device__ UInt2 operator += (const UInt2& v)      { x += v.x; y += v.y; return *this; }
    inline __host__ __device__ UInt2 operator -= (const UInt2& v)      { x -= v.x; y -= v.y; return *this; }

    inline __host__ __device__ bool operator == (const UInt2& v)      { return x == v.x && y == v.y; }
	inline __host__ __device__ bool operator != (const UInt2& v)      { return x != v.x || y != v.y; }

	inline __host__ __device__ unsigned int& operator[] (unsigned int i)               { return _v[i]; }
	inline __host__ __device__ unsigned int  operator[] (unsigned int i) const         { return _v[i]; }

	inline __host__ __device__ operator Int2() const { return Int2((int)x, (int)y); }
};

// Int2 - Float2

// Int2 operator * (const Float2& v, int a) { return Int2(v.x * a, v.y * a); }
// Int2 operator / (const Float2& v, int a) { return Int2(v.x / a, v.y / a); }

inline __host__ __device__ Int2 operator + (int a, const Int2& v)   { return Int2(v.x + a, v.y + a); }
inline __host__ __device__ Int2 operator - (int a, const Int2& v)   { return Int2(a - v.x, a - v.y); }
inline __host__ __device__ Int2 operator * (int a, const Int2& v)   { return Int2(v.x * a, v.y * a); }
inline __host__ __device__ Int2 operator / (int a, const Int2& v)   { return Int2(a / v.x, a / v.y); }

inline __host__ __device__ Float2 operator + (float a, const Int2& v) { return Float2((float)v.x + a, (float)v.y + a); }
inline __host__ __device__ Float2 operator - (float a, const Int2& v) { return Float2(a - (float)v.x, a - (float)v.y); }
inline __host__ __device__ Float2 operator * (float a, const Int2& v) { return Float2((float)v.x * a, (float)v.y * a); }
inline __host__ __device__ Float2 operator / (float a, const Int2& v) { return Float2(a / (float)v.x, a / (float)v.y); }

inline __host__ __device__ Float2 operator + (const Float2& vf, const Int2& vi) { return Float2(vf.x + vi.x, vf.y + vi.y); }
inline __host__ __device__ Float2 operator - (const Float2& vf, const Int2& vi) { return Float2(vf.x - vi.x, vf.y - vi.y); }
inline __host__ __device__ Float2 operator * (const Float2& vf, const Int2& vi) { return Float2(vf.x * vi.x, vf.y * vi.y); }
inline __host__ __device__ Float2 operator / (const Float2& vf, const Int2& vi) { return Float2(vf.x / vi.x, vf.y / vi.y); }

inline __host__ __device__ Float2 operator + (const Int2& vi, const Float2& vf) { return Float2(vi.x + vf.x, vi.y + vf.y); }
inline __host__ __device__ Float2 operator - (const Int2& vi, const Float2& vf) { return Float2(vi.x - vf.x, vi.y - vf.y); }
inline __host__ __device__ Float2 operator * (const Int2& vi, const Float2& vf) { return Float2(vi.x * vf.x, vi.y * vf.y); }
inline __host__ __device__ Float2 operator / (const Int2& vi, const Float2& vf) { return Float2(vi.x / vf.x, vi.y / vf.y); }

inline __device__ Float2 floor (const Float2& v) { return Float2(floorf(v.x), floorf(v.y)); }
inline __device__ Int2   floori(const Float2& v) { return Int2((int)(floorf(v.x)), (int)(floorf(v.y))); }
inline __device__ Float2 fract (const Float2& v) { float intPart; return Float2(modff(v.x, &intPart), modff(v.y, &intPart)); }
inline __device__ Int2   roundi(const Float2& v) { return Int2((int)(rintf(v.x)), (int)(rintf(v.y))); }

inline __host__ __device__ float max1f(const float& a, const float& b){ return (a < b) ? b : a; }
inline __host__ __device__ float min1f(const float& a, const float& b){ return (a > b) ? b : a; }

struct Float3
{
	union {
		struct { float x, y, z; };
		float _v[3];
	};

	__host__ __device__ Float3()                             : x(0), y(0), z(0)       {}
	__host__ __device__ Float3(float _x)                     : x(_x), y(_x), z(_x)    {}
	__host__ __device__ Float3(float _x, float _y, float _z) : x(_x), y(_y), z(_z)    {}

	inline __host__ __device__ Float3  operator+(const Float3& v) const { return Float3(x + v.x, y + v.y, z + v.z); }
	inline __host__ __device__ Float3  operator-(const Float3& v) const { return Float3(x - v.x, y - v.y, z - v.z); }
	inline __host__ __device__ Float3  operator*(const Float3& v) const { return Float3(x * v.x, y * v.y, z * v.z); }
	inline __host__ __device__ Float3  operator/(const Float3& v) const { return Float3(x / v.x, y / v.y, z / v.z); }

	inline __host__ __device__ Float3  operator+(float a) const         { return Float3(x + a, y + a, z + a); }
	inline __host__ __device__ Float3  operator-(float a) const         { return Float3(x - a, y - a, z - a); }
	inline __host__ __device__ Float3  operator*(float a) const         { return Float3(x * a, y * a, z * a); }
	inline __host__ __device__ Float3  operator/(float a) const         { return Float3(x / a, y / a, z / a); }

	inline __host__ __device__ Float3& operator+=(const Float3& v)      { x += v.x; y += v.y; z += v.z; return *this; }
	inline __host__ __device__ Float3& operator-=(const Float3& v)      { x -= v.x; y -= v.y; z -= v.z; return *this; }
	inline __host__ __device__ Float3& operator*=(const Float3& v)      { x *= v.x; y *= v.y; z *= v.z; return *this; }
	inline __host__ __device__ Float3& operator/=(const Float3& v)      { x /= v.x; y /= v.y; z /= v.z; return *this; }

	inline __host__ __device__ Float3& operator+=(const float& a)       { x += a; y += a; z += a; return *this; }
	inline __host__ __device__ Float3& operator-=(const float& a)       { x -= a; y -= a; z -= a; return *this; }
	inline __host__ __device__ Float3& operator*=(const float& a)       { x *= a; y *= a; z *= a; return *this; }
	inline __host__ __device__ Float3& operator/=(const float& a)       { x /= a; y /= a; z /= a; return *this; }

	inline __host__ __device__ Float3 operator-() const { return Float3(-x, -y, -z); }

	inline __host__ __device__ bool operator!=(const Float3& v) const   { return x != v.x || y != v.y || z != v.z; }
	inline __host__ __device__ bool operator==(const Float3& v) const   { return x == v.x && y == v.y && z == v.z; }

	inline __host__ __device__ float& operator[](int i)                 { return _v[i]; }
	inline __host__ __device__ float  operator[](int i) const           { return _v[i]; }

	inline __host__ __device__ float   length() const                   { return sqrtf(x*x + y*y + z*z); }
	inline __host__ __device__ float   length2() const                  { return x*x + y*y + z*z; }
	inline __host__ __device__ float   max() const                      { return max1f(max1f(x, y), z); }
	inline __host__ __device__ float   min() const                      { return min1f(min1f(x, y), z); }
	inline __host__ __device__ Float3& normalize()                      { float norm = sqrtf(x*x + y*y + z*z); x /= norm; y /= norm; z /= norm; return *this; }
	inline __host__ __device__ Float3  normalized() const               { float norm = sqrtf(x*x + y*y + z*z); return Float3(x / norm, y / norm, z / norm); }
};

inline __host__ __device__ Float3 operator+(float a, const Float3& v)   { return Float3(v.x + a, v.y + a, v.z + a); }
inline __host__ __device__ Float3 operator-(float a, const Float3& v)   { return Float3(a - v.x, a - v.y, a - v.z); }
inline __host__ __device__ Float3 operator*(float a, const Float3& v)   { return Float3(v.x * a, v.y * a, v.z * a); }
inline __host__ __device__ Float3 operator/(float a, const Float3& v)   { return Float3(a / v.x, a / v.y, a / v.z); }

struct Int3
{
	union {
		struct { int x, y, z; };
		int _v[3];
	};

	__host__ __device__ Int3()                     : x{0}, y{0}, z{0} {}
	__host__ __device__ Int3(int a)                : x{a}, y{a}, z{a} {}
    __host__ __device__ Int3(int x, int y, int z)  : x{x}, y{y}, z{z} {}

    inline __host__ __device__ Int3 operator + (int a) const         { return Int3(x + a, y + a, z + a); }
    inline __host__ __device__ Int3 operator - (int a) const         { return Int3(x - a, y - a, z - a); }

	inline __host__ __device__ Int3 operator += (int a)              { x += a; y += a; z += a; return *this; }
    inline __host__ __device__ Int3 operator -= (int a)              { x -= a; y -= a; z -= a; return *this; }

    inline __host__ __device__ Int3 operator + (const Int3& v) const { return Int3(x + v.x, y + v.y, z + v.y); }
    inline __host__ __device__ Int3 operator - (const Int3& v) const { return Int3(x - v.x, y - v.y, z - v.y); }

	inline __host__ __device__ Int3 operator += (const Int3& v)      { x += v.x; y += v.y; z += v.y; return *this; }
    inline __host__ __device__ Int3 operator -= (const Int3& v)      { x -= v.x; y -= v.y; z -= v.y; return *this; }

	inline __host__ __device__ Int3 operator - () const              { return Int3(-x, -y, -z); }

    inline __host__ __device__ bool operator == (const Int3& v)      { return x == v.x && y == v.y && z == v.z; }
	inline __host__ __device__ bool operator != (const Int3& v)      { return x != v.x || y != v.y || z != v.z; }

	inline __host__ __device__ int& operator[] (int i)               { return _v[i]; }
	inline __host__ __device__ int  operator[] (int i) const         { return _v[i]; }
};

struct Float4
{
	union {
		struct { float x, y, z, w; };
		struct { Float3 xyz; float w; };
		float _v[4];
	};

	__host__ __device__ Float4()                                       : x(0), y(0), z(0), w(0)             {}
	__host__ __device__ Float4(float _x)                               : x(_x), y(_x), z(_x), w(_x)         {}
	__host__ __device__ Float4(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w)         {}
	__host__ __device__ Float4(const Float2& v1, const Float2& v2)     : x(v1.x), y(v1.y), z(v2.x), w(v2.y) {}
	__host__ __device__ Float4(const Float3& v)                        : x(v.x), y(v.y), z(v.z), w(0)       {}
	__host__ __device__ Float4(const Float3& v, float a)               : x(v.x), y(v.y), z(v.z), w(a)       {}

	inline __host__ __device__ Float4  operator+(const Float4& v) const { return Float4(x + v.x, y + v.y, z + v.z, z + v.z); }
	inline __host__ __device__ Float4  operator-(const Float4& v) const { return Float4(x - v.x, y - v.y, z - v.z, z - v.z); }
	inline __host__ __device__ Float4  operator*(const Float4& v) const { return Float4(x * v.x, y * v.y, z * v.z, z * v.z); }
	inline __host__ __device__ Float4  operator/(const Float4& v) const { return Float4(x / v.x, y / v.y, z / v.z, z / v.z); }

	inline __host__ __device__ Float4  operator+(float a) const         { return Float4(x + a, y + a, z + a, z + a); }
	inline __host__ __device__ Float4  operator-(float a) const         { return Float4(x - a, y - a, z - a, z - a); }
	inline __host__ __device__ Float4  operator*(float a) const         { return Float4(x * a, y * a, z * a, z * a); }
	inline __host__ __device__ Float4  operator/(float a) const         { return Float4(x / a, y / a, z / a, z / a); }

	inline __host__ __device__ Float4& operator+=(const Float4& v)      { x += v.x; y += v.y; z += v.z; w += v.w; return *this; }
	inline __host__ __device__ Float4& operator-=(const Float4& v)      { x -= v.x; y -= v.y; z -= v.z; w -= v.w; return *this; }
	inline __host__ __device__ Float4& operator*=(const Float4& v)      { x *= v.x; y *= v.y; z *= v.z; w *= v.w; return *this; }
	inline __host__ __device__ Float4& operator/=(const Float4& v)      { x /= v.x; y /= v.y; z /= v.z; w /= v.w; return *this; }

	inline __host__ __device__ Float4& operator+=(const float& a)       { x += a; y += a; z += a; w += a; return *this; }
	inline __host__ __device__ Float4& operator-=(const float& a)       { x -= a; y -= a; z -= a; w += a; return *this; }
	inline __host__ __device__ Float4& operator*=(const float& a)       { x *= a; y *= a; z *= a; w += a; return *this; }
	inline __host__ __device__ Float4& operator/=(const float& a)       { x /= a; y /= a; z /= a; w += a; return *this; }

	inline __host__ __device__ Float4 operator-() const { return Float4(-x, -y, -z, -w); }

	inline __host__ __device__ bool operator!=(const Float4& v) const   { return x != v.x || y != v.y || z != v.z || w != v.w; }
	inline __host__ __device__ bool operator==(const Float4& v) const   { return x == v.x && y == v.y && z == v.z && w == v.w; }

	inline __host__ __device__ float& operator[](int i)                 { return _v[i]; }
	inline __host__ __device__ float  operator[](int i) const           { return _v[i]; }
};

inline __host__ __device__ Float4 operator + (float a, const Float4& v) { return Float4(v.x + a, v.y + a, v.z + a, v.w + a); }
inline __host__ __device__ Float4 operator - (float a, const Float4& v) { return Float4(a - v.x, a - v.y, a - v.z, a - v.w); }
inline __host__ __device__ Float4 operator * (float a, const Float4& v) { return Float4(v.x * a, v.y * a, v.z * a, v.w * a); }
inline __host__ __device__ Float4 operator / (float a, const Float4& v) { return Float4(a / v.x, a / v.y, a / v.z, a / v.w); }

// struct Int4
// {
// 	int x, y, z, w;

// 	__host__ __device__ Int4(int _x = 0, int _y = 0, int _z = 0, int _w = 0) : x(_x), y(_y), z(_z), w(_w) {}
// 	__host__ __device__ Int4(const Int4& v) : x(v.x), y(v.y), z(v.z), w(v.w) {}
// 	__host__ __device__ Int4(const Int3& v, const int a) : x(v.x), y(v.y), z(v.z), w(a) {}
// };

inline __host__ __device__ Float3 abs(const Float3& v)                                { return Float3(fabsf(v.x), fabsf(v.y), fabsf(v.z)); }
inline __host__ __device__ Float3 max(const Float3& v1, const Float3& v2)             { return Float3(max1f(v1.x, v2.x), max1f(v1.y, v2.y), max1f(v1.z, v2.z)); }
inline __host__ __device__ Float3 normalize(const Float3& v)                          { float norm = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z); return Float3(v.x / norm, v.y / norm, v.z / norm); }
inline __host__ __device__ Float3 sqrt3f(const Float3& v)                             { return Float3(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z)); }
inline __host__ __device__ Float3 min3f(const Float3 & v1, const Float3 & v2)         { return Float3(v1.x < v2.x ? v1.x : v2.x, v1.y < v2.y ? v1.y : v2.y, v1.z < v2.z ? v1.z : v2.z); }
inline __host__ __device__ Float3 max3f(const Float3 & v1, const Float3 & v2)         { return Float3(v1.x > v2.x ? v1.x : v2.x, v1.y > v2.y ? v1.y : v2.y, v1.z > v2.z ? v1.z : v2.z); }
inline __host__ __device__ Float3 cross(const Float3 & v1, const Float3 & v2)         { return Float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x); }
inline __host__ __device__ Float3 powf(const Float3 & v1, const Float3 & v2)          { return Float3(powf(v1.x, v2.x), powf(v1.y, v2.y), powf(v1.z, v2.z)); }
inline __host__ __device__ Float3 exp3f(const Float3 & v)                             { return Float3(expf(v.x), expf(v.y), expf(v.z)); }
inline __host__ __device__ Float3 pow3f(const Float3& v, float a)                     { return Float3(powf(v.x, a), powf(v.y, a), powf(v.z, a)); }
inline __host__ __device__ Float3 mixf(const Float3 & v1, const Float3 & v2, float a) { return v1 * (1.0f - a) + v2 * a; }
inline __host__ __device__ Float3 minf3f(const float a, const Float3 & v)             { return Float3(v.x < a ? v.x : a, v.y < a ? v.y : a, v.y < a ? v.y : a); }
inline __host__ __device__ void   swap(float& v1, float& v2)                          { float tmp = v1; v1 = v2; v2 = tmp; }
inline __host__ __device__ void   swap(Float3 & v1, Float3 & v2)                      { Float3 tmp = v1; v1 = v2; v2 = tmp; }
inline __host__ __device__ float  clampf(float a, float lo, float hi)                 { return a < lo ? lo : a > hi ? hi : a; }
inline __host__ __device__ Float3 clamp3f(Float3 a, Float3 lo, Float3 hi)             { return Float3(clampf(a.x, lo.x, hi.x), clampf(a.y, lo.y, hi.y), clampf(a.z, lo.z, hi.z)); }

//inline __host__ __device__ float  smoothstep(float edge0, float edge1, float x)       { float t; t = clampf((x - edge0) / (edge1 - edge0), 0.0f, 1.0f); return t * t * (3.0f - 2.0f * t); }
inline __host__ __device__ float  dot(const Float3 & v1, const Float3 & v2)           { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }
inline __host__ __device__ float  dot(const Float4 & v1, const Float4 & v2)           { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w; }
inline __host__ __device__ float  distancesq(const Float3 & v1, const Float3 & v2)    { return (v1.x - v2.x) * (v1.x - v2.x) + (v1.y - v2.y) * (v1.y - v2.y) + (v1.z - v2.z) * (v1.z - v2.z); }
inline __host__ __device__ float  distance(const Float3 & v1, const Float3 & v2)      { return sqrtf((v1.x - v2.x) * (v1.x - v2.x) + (v1.y - v2.y) * (v1.y - v2.y) + (v1.z - v2.z) * (v1.z - v2.z)); }
inline __host__ __device__ Float3 lerp3f(Float3 a, Float3 b, float w) { return a + w * (b - a); }

inline __host__ __device__ unsigned int divRoundUp(unsigned int dividend, unsigned int divisor) { return (dividend + divisor - 1) / divisor; }

// struct Mat4
// {
// 	union {
// 		struct {
// 			float m00, m10, m20, m30;
// 			float m01, m11, m21, m31;
// 			float m02, m12, m22, m32;
// 			float m03, m13, m23, m33;
// 		};
// 		struct {
// 			Float4 v0, v1, v2, v3;
// 		};
// 		Float4 _v4[4];
// 		float _v[16];
// 	};
// 	Mat4() { for (int i = 0; i < 16; ++i) _v[i] = 0; }
// 	Mat4(const Float4& v0, const Float4& v1, const Float4& v2, const Float4& v3) : v0{v0}, v1{v1}, v2{v2}, v3{v3} {}
// 	Mat4(const Mat4& other) { for (int i = 0; i < 16; ++i) _v[i] = other._v[i]; }

// 	inline __host__ __device__ Float4&       operator[](int i)       { return _v4[i]; }
// 	inline __host__ __device__ const Float4& operator[](int i) const { return _v4[i]; }

// 	inline __host__ __device__ Float4 GetColumn(int i) const { return Float4(v0[i], v1[i], v2[i], v3[i]); }
// };

// inline __host__ __device__ Float4 operator*(const Float4& v, const Mat4& m) { return Float4(dot(m.GetColumn(0), v), dot(m.GetColumn(1), v), dot(m.GetColumn(2), v), dot(m.GetColumn(3), v)); }
// inline __host__ __device__ Float4 operator*(const Mat4& m, const Float4& v) { return Float4(dot(m.v0, v), dot(m.v1, v), dot(m.v2, v), dot(m.v3, v)); }

struct Quat
{
	union {
		Float3 v;
		struct {
			float x, y, z;
		};
	};
	float w;

	__host__ __device__ Quat() : v(), w(0) {}
	__host__ __device__ Quat(const Float3& v) : v(v), w(0) {}
	__host__ __device__ Quat(const Float3& v, float w) : v(v), w(w) {}
	__host__ __device__ Quat(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}

	static inline __host__ __device__ Quat axisAngle(const Float3& axis, float angle) { return Quat(axis.normalized() * sinf(angle / 2), cosf(angle / 2)); }

	inline __host__ __device__ Quat  conj      () const        { return Quat(-v, w); }
	inline __host__ __device__ float norm2     () const        { return x*x + y*y + z*z + w*w; }
	inline __host__ __device__ Quat  inv       () const        { return conj() / norm2(); }
	inline __host__ __device__ float norm      () const        { return sqrtf(norm2()); }
	inline __host__ __device__ Quat  normalized() const        { float n = norm(); return Quat(v / n, w / n); }
	inline __host__ __device__ Quat  pow       (float a) const { return Quat::axisAngle(v, acosf(w) * a * 2);  }

	inline __host__ __device__ Quat operator/  (float a) const        { return Quat(v / a, w / a); }
	inline __host__ __device__ Quat operator+  (const Quat& q) const  { const Quat& p = *this; return Quat (p.v + q.v, p.w + q.w); }
	inline __host__ __device__ Quat operator*  (const Quat & q) const { const Quat& p = *this; return Quat(p.w * q.v + q.w * p.v + cross(p.v, q.v), p.w * q.w - dot(p.v, q.v)); }
	inline __host__ __device__ Quat& operator+=(const Quat& q)        { Quat ret = *this + q; return (*this = ret); }
	inline __host__ __device__ Quat& operator*=(const Quat& q)        { Quat ret = *this * q; return (*this = ret); }
};

inline __host__ __device__ Quat rotate         (const Quat& q, const Quat& v)                         { return q * v * q.conj(); }
inline __host__ __device__ Quat slerp          (const Quat& q, const Quat& r, float t)                { return (r * q.conj()).pow(t) * q; }
inline __host__ __device__ Quat rotationBetween(const Quat& p, const Quat& q)                         { return Quat(cross(p.v, q.v), sqrtf(p.v.length2() * q.v.length2()) + dot(p.v, q.v)).normalized(); }

inline __host__ __device__ Float3 rotate3f         (const Float3& axis, float angle, const Float3& v) { return rotate(Quat::axisAngle(axis, angle), v).v; }
inline __host__ __device__ Float3 slerp3f          (const Float3& q, const Float3& r, float t)        { return slerp(Quat(q), Quat(r), t).v; }
inline __host__ __device__ Float3 rotationBetween3f(const Float3& p, const Float3& q)                 { return rotationBetween(Quat(p), Quat(q)).v; }
