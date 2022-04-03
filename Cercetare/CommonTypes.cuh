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

struct AABB {
	glm::vec3 min;
	glm::vec3 max;
};

struct Node {
	AABB aabb;
	Node* parent;
	Node* leftNode;
	Node* rightNode;
	unsigned int idx;
	unsigned int objectID;
	NodeType nodeType;
};