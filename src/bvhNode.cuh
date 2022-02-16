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

__host__ __device__ inline void Print(const char* name, const BVHNode& node)
{
	const AABBCompact& aabb = node.aabb;
	printf("%s = BVHNode {\n"
	       "    AABBCompact {\n"
		   "        left AABB min=(%f, %f, %f), max=(%f, %f, %f);\n"
		   "        right AABB min=(%f, %f, %f), max=(%f, %f, %f);\n"
		   "    }\n"
		   "    idxLeft = %d, idxRight = %d;\n"
		   "    isLeftLeaf = %d, isRightLeaf = %d;\n"
		   "}\n",
	name, aabb.GetLeftAABB().min[0], aabb.GetLeftAABB().min[1], aabb.GetLeftAABB().min[2], aabb.GetLeftAABB().max[0], aabb.GetLeftAABB().max[1], aabb.GetLeftAABB().max[2]
	, aabb.GetRightAABB().min[0], aabb.GetRightAABB().min[1], aabb.GetRightAABB().min[2], aabb.GetRightAABB().max[0], aabb.GetRightAABB().max[1], aabb.GetRightAABB().max[2],
	node.idxLeft, node.idxRight, node.isLeftLeaf, node.isRightLeaf);
}