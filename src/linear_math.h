#pragma once

#include <math.h>
#include <cuda_runtime.h>

#define uint unsigned int
#define ushort unsigned short
#define ullint unsigned long long int
#define ushort unsigned short

#define PI_OVER_4               0.7853981633974483096156608458198757210492f
#define PI_OVER_2               1.5707963267948966192313216916397514420985f
#define SQRT_OF_ONE_THIRD       0.5773502691896257645091487805019574556476f
#ifndef M_PI
#define M_PI                    3.1415926535897932384626422832795028841971f
#endif // !M_PI
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

	inline __host__ __device__ float   length() const                   { return sqrtf(x*x + y*y); }
	inline __host__ __device__ float   length2() const                  { return x*x + y*y; }

};

inline __host__ __device__ Float2 operator+(float a, const Float2& v)   { return Float2(v.x + a, v.y + a); }
inline __host__ __device__ Float2 operator-(float a, const Float2& v)   { return Float2(a - v.x, a - v.y); }
inline __host__ __device__ Float2 operator*(float a, const Float2& v)   { return Float2(v.x * a, v.y * a); }
inline __host__ __device__ Float2 operator/(float a, const Float2& v)   { return Float2(a / v.x, a / v.y); }

inline __host__ __device__ float length(const Float2& v) { return sqrtf(v.x * v.x + v.y * v.y); }

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

inline __device__ float  fract (float a) { float intPart; return modff(a, &intPart); }
inline __device__ Float2 floor (const Float2& v) { return Float2(floorf(v.x), floorf(v.y)); }
inline __device__ Int2   floori(const Float2& v) { return Int2((int)(floorf(v.x)), (int)(floorf(v.y))); }
inline __host__   Int2   floor2(const Float2& v) { return Int2((int)(floor(v.x)), (int)(floor(v.y))); }
inline __device__ Float2 fract (const Float2& v) { float intPart; return Float2(modff(v.x, &intPart), modff(v.y, &intPart)); }
inline __device__ Int2   roundi(const Float2& v) { return Int2((int)(rintf(v.x)), (int)(rintf(v.y))); }

inline __host__ __device__ float max1f(const float& a, const float& b){ return (a < b) ? b : a; }
inline __host__ __device__ float min1f(const float& a, const float& b){ return (a > b) ? b : a; }

struct Float3
{
	union {
		struct { float x, y, z; };
		struct { Float2 xy; float z; };
		float _v[3];
	};

	__host__ __device__ Float3()                             : x(0), y(0), z(0)       {}
	__host__ __device__ Float3(float _x)                     : x(_x), y(_x), z(_x)    {}
	__host__ __device__ Float3(float _x, float _y, float _z) : x(_x), y(_y), z(_z)    {}

	inline __host__ __device__ Float2 xz() const { return Float2(x, z); }

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
	inline __host__ __device__ float   getmax() const                   { return max(max(x, y), z); }
	inline __host__ __device__ float   getmin() const                   { return min(min(x, y), z); }
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
		struct { Float2 xy; Float2 zw; };
		float _v[4];
	};

	__host__ __device__ Float4()                                       : x(0), y(0), z(0), w(0)             {}
	__host__ __device__ Float4(float _x)                               : x(_x), y(_x), z(_x), w(_x)         {}
	__host__ __device__ Float4(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w)         {}
	__host__ __device__ Float4(const Float2& v1, const Float2& v2)     : x(v1.x), y(v1.y), z(v2.x), w(v2.y) {}
	__host__ __device__ Float4(const Float3& v)                        : x(v.x), y(v.y), z(v.z), w(0)       {}
	__host__ __device__ Float4(const Float3& v, float a)               : x(v.x), y(v.y), z(v.z), w(a)       {}
	__host__ __device__ Float4(const Float4& v)                        : x(v.x), y(v.y), z(v.z), w(v.w)     {}

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
inline __host__ __device__ Float2 normalize(const Float2& v)                          { float norm = sqrtf(v.x * v.x + v.y * v.y); return Float2(v.x / norm, v.y / norm); }
inline __host__ __device__ Float3 normalize(const Float3& v)                          { float norm = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z); return Float3(v.x / norm, v.y / norm, v.z / norm); }
inline __host__ __device__ Float3 sqrt3f(const Float3& v)                             { return Float3(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z)); }
inline __host__ __device__ Float3 rsqrt3f(const Float3& v)                            { return Float3(rsqrtf(v.x), rsqrtf(v.y), rsqrtf(v.z)); }
inline __host__ __device__ Float3 min3f(const Float3 & v1, const Float3 & v2)         { return Float3(min(v1.x, v2.x), min(v1.y, v2.y), min(v1.z, v2.z)); }
inline __host__ __device__ Float3 max3f(const Float3 & v1, const Float3 & v2)         { return Float3(max(v1.x, v2.x), max(v1.y, v2.y), max(v1.z, v2.z)); }
inline __host__ __device__ Float3 cross(const Float3 & v1, const Float3 & v2)         { return Float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x); }
inline __host__ __device__ Float3 powf(const Float3 & v1, const Float3 & v2)          { return Float3(powf(v1.x, v2.x), powf(v1.y, v2.y), powf(v1.z, v2.z)); }
inline __host__ __device__ Float3 exp3f(const Float3 & v)                             { return Float3(expf(v.x), expf(v.y), expf(v.z)); }
inline __host__ __device__ Float3 pow3f(const Float3& v, float a)                     { return Float3(powf(v.x, a), powf(v.y, a), powf(v.z, a)); }
inline __host__ __device__ Float3 sin3f(const Float3 & v)                             { return Float3(sinf(v.x), sinf(v.y), sinf(v.z)); }
inline __host__ __device__ Float3 cos3f(const Float3 & v)                             { return Float3(cosf(v.x), cosf(v.y), cosf(v.z)); }
inline __host__ __device__ Float3 mixf(const Float3 & v1, const Float3 & v2, float a) { return v1 * (1.0f - a) + v2 * a; }
inline __host__ __device__ float  mix1f(float v1, float v2, float a)                  { return v1 * (1.0f - a) + v2 * a; }
inline __host__ __device__ Float3 minf3f(const float a, const Float3 & v)             { return Float3(v.x < a ? v.x : a, v.y < a ? v.y : a, v.y < a ? v.y : a); }
inline __host__ __device__ void   swap(float& v1, float& v2)                          { float tmp = v1; v1 = v2; v2 = tmp; }
inline __host__ __device__ void   swap(Float3 & v1, Float3 & v2)                      { Float3 tmp = v1; v1 = v2; v2 = tmp; }
inline __host__ __device__ float  clampf(float a, float lo = 0.0f, float hi = 1.0f)   { return a < lo ? lo : a > hi ? hi : a; }
inline __host__ __device__ Float3 clamp3f(Float3 a, Float3 lo = Float3(0.0f), Float3 hi = Float3(1.0f)){ return Float3(clampf(a.x, lo.x, hi.x), clampf(a.y, lo.y, hi.y), clampf(a.z, lo.z, hi.z)); }
inline __host__ __device__ float  smoothstep1f(float edge0, float edge1, float x)     { float t; t = clampf((x - edge0) / (edge1 - edge0), 0.0f, 1.0f); return t * t * (3.0f - 2.0f * t); }
inline __host__ __device__ float  dot(const Float2 & v1, const Float2 & v2)           { return v1.x * v2.x + v1.y * v2.y; }
inline __host__ __device__ float  dot(const Float3 & v1, const Float3 & v2)           { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }
inline __host__ __device__ float  dot(const Float4 & v1, const Float4 & v2)           { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w; }
inline __host__ __device__ float  distancesq(const Float3 & v1, const Float3 & v2)    { return (v1.x - v2.x) * (v1.x - v2.x) + (v1.y - v2.y) * (v1.y - v2.y) + (v1.z - v2.z) * (v1.z - v2.z); }
inline __host__ __device__ float  distance(const Float3 & v1, const Float3 & v2)      { return sqrtf((v1.x - v2.x) * (v1.x - v2.x) + (v1.y - v2.y) * (v1.y - v2.y) + (v1.z - v2.z) * (v1.z - v2.z)); }
inline __host__ __device__ Float3 lerp3f(Float3 a, Float3 b, float w)                 { return a + w * (b - a); }
inline __host__ __device__ float  lerpf(float a, float b, float w)                    { return a + w * (b - a); }
inline __host__ __device__ Float3 reflect3f(Float3 i, Float3 n)                       { return i - 2.0f * n * dot(n,i); }
inline __host__ __device__ float  pow2(float a)                                       { return a * a; }
inline __host__ __device__ float  pow3(float a)                                       { return a * a * a; }

inline __host__ __device__ unsigned int divRoundUp(unsigned int dividend, unsigned int divisor) { return (dividend + divisor - 1) / divisor; }

struct Mat3
{
	union {
		struct {
			float m00, m10, m20;
			float m01, m11, m21;
			float m02, m12, m22;
		};
		struct {
			Float3 v0, v1, v2;
		};
		Float3 _v3[3];
		float _v[9];
	};
	__host__ __device__ Mat3() { for (int i = 0; i < 9; ++i) _v[i] = 0; }
	__host__ __device__ Mat3(const Float3& v0, const Float3& v1, const Float3& v2) : v0{v0}, v1{v1}, v2{v2} {}

	inline __host__ __device__ Float3&       operator[](int i)       { return _v3[i]; }
	inline __host__ __device__ const Float3& operator[](int i) const { return _v3[i]; }

	inline __host__ __device__ void transpose() { swap(m01, m10); swap(m20, m02); swap(m21, m12); }
};

// column major multiply
inline __host__ __device__ Float3 operator*(const Mat3& m, const Float3& v) { return v.x * m.v0 +  v.y * m.v1 +  v.z * m.v2; }

// Rotation matrix
inline __host__ __device__ Mat3 RotationMatrixX(float a) { return { Float3(1, 0, 0), Float3(0, cosf(a), sinf(a)), Float3(0, -sinf(a), cosf(a)) }; }
inline __host__ __device__ Mat3 RotationMatrixY(float a) { return { Float3(cosf(a), 0, -sinf(a)), Float3(0, 1, 0), Float3(sinf(a), 0, cosf(a)) }; }
inline __host__ __device__ Mat3 RotationMatrixZ(float a) { return { Float3(cosf(a), sinf(a), 0), Float3(-sinf(a), cosf(a), 0), Float3(0, 0, 1) }; }

struct Mat4
{
	union {
		struct {
			float m00, m10, m20, m30;
			float m01, m11, m21, m31;
			float m02, m12, m22, m32;
			float m03, m13, m23, m33;
		};
		float _v[16];
	};

	__host__ __device__ Mat4() { for (int i = 0; i < 16; ++i) { _v[i] = 0; } m00 = m11 = m22 = m33 = 1; }
	__host__ __device__ Mat4(const Mat4& m) { for (int i = 0; i < 16; ++i) { _v[i] = m[i]; } }

	// row
	inline __host__ __device__ void          setRow(uint i, const Float4& v)       { /*assert(i < 4);*/ _v[i] = v[0]; _v[i+4] = v[1]; _v[i+8] = v[2]; _v[i+12] = v[3]; }
	inline __host__ __device__ Float4        getRow(uint i) const                  { /*assert(i < 4);*/ return Float4(_v[i], _v[i+4], _v[i+8], _v[i+12]); }

	// column
	inline __host__ __device__ void          setCol(uint i, const Float4& v)       { /*assert(i < 4);*/ _v[i*4] = v[0]; _v[i*4+1] = v[1]; _v[i*4+2] = v[2]; _v[i*4+3] = v[3]; }
	inline __host__ __device__ Float4        getCol(uint i) const                  { /*assert(i < 4);*/ return Float4(_v[i*4], _v[i*4+1], _v[i*4+2], _v[i*4+3]); }

	// element
	inline __host__ __device__ void          set(uint r, uint c, float v)          { /*assert(r < 4 && c < 4);*/ _v[r + c * 4] = v; }
	inline __host__ __device__ float         get(uint r, uint c) const             { /*assert(r < 4 && c < 4);*/ return _v[r + c * 4]; }

	inline __host__ __device__ float         operator[](uint i) const { return _v[i]; }
	inline __host__ __device__ float&        operator[](uint i)       { return _v[i]; }
};

inline __host__ __device__ Mat4 invert(const Mat4& m)
{
	Mat4 inv;

	inv[0]  =  m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
	inv[4]  = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
	inv[8]  =  m[4] * m[9] *  m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
	inv[12] = -m[4] * m[9] *  m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
	inv[1]  = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
	inv[5]  =  m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
	inv[9]  = -m[0] * m[9] *  m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
	inv[13] =  m[0] * m[9] *  m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
	inv[2]  =  m[1] * m[6] *  m[15] - m[1] * m[7] *  m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] -  m[13] * m[3] * m[6];
	inv[6]  = -m[0] * m[6] *  m[15] + m[0] * m[7] *  m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] +  m[12] * m[3] * m[6];
	inv[10] =  m[0] * m[5] *  m[15] - m[0] * m[7] *  m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] -  m[12] * m[3] * m[5];
	inv[14] = -m[0] * m[5] *  m[14] + m[0] * m[6] *  m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] +  m[12] * m[2] * m[5];
	inv[3]  = -m[1] * m[6] *  m[11] + m[1] * m[7] *  m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] *  m[2] * m[7] +  m[9] *  m[3] * m[6];
	inv[7]  =  m[0] * m[6] *  m[11] - m[0] * m[7] *  m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] *  m[2] * m[7] -  m[8] *  m[3] * m[6];
	inv[11] = -m[0] * m[5] *  m[11] + m[0] * m[7] *  m[9] +  m[4] * m[1] * m[11] - m[4] * m[3] * m[9] -  m[8] *  m[1] * m[7] +  m[8] *  m[3] * m[5];
	inv[15] =  m[0] * m[5] *  m[10] - m[0] * m[6] *  m[9] -  m[4] * m[1] * m[10] + m[4] * m[2] * m[9] +  m[8] *  m[1] * m[6] -  m[8] *  m[2] * m[5];

	float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

	if (det == 0) { return Mat4(); }
	det = 1.f / det;
	for (int i = 0; i < 16; i++) { inv[i] *= det; }

	return inv;
}

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

__host__ __device__ __inline__ float SafeDivide(float a, float b) { float eps = exp2f(-80.0f); return a / ((fabsf(b) > eps) ? b : copysignf(eps, b)); };
__host__ __device__ __inline__ Float3 SafeDivide3f(const Float3& a, const Float3& b) { return Float3(SafeDivide(a.x, b.x), SafeDivide(a.y, b.y), SafeDivide(a.z, b.z)); };