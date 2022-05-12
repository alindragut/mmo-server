#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#define GLM_FORCE_CUDA
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "Memory.cuh"

enum NodeType {
	Root,
	Internal,
	Leaf
};

enum ShapeType {
	None,
	SphereShape,
	BoxShape,
	CapsuleShape
};

struct Sphere {
	float radius;
};

struct Box {
	glm::vec3 halfExtents;
};

struct Capsule {
	float halfHeight;
	float radius;
};

struct AABB {
	glm::vec3 min;
	glm::vec3 max;
};

typedef struct {
	ShapeType type;
	union {
		Sphere sphere;
		Box box;
		Capsule capsule;
	};
} Shape;

struct Node {
	AABB aabb;
	Node* parent;
	Node* leftNode;
	Node* rightNode;
	unsigned int idx;
	unsigned int objectID;
	NodeType nodeType;
};

struct CollisionManifold {
	glm::vec3 collisionNormal;
	float depth;
	float nrContactPoints;
};