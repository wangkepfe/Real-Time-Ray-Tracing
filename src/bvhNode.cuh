#pragma once

#include "geometry.h"

struct __align__(16) BVHNode
{
	AABBCompact aabb;

	uint idxLeft;
	uint idxRight;
	uint isLeftLeaf;
	uint isRightLeaf;
};