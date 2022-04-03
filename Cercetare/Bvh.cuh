#pragma once
#include "CommonTypes.cuh"

class BVH {
public:
	BVH();
	~BVH();

	void Build(glm::vec3* d_positions, int arraySize);

	void BroadPhase(glm::vec3* d_positions, int arraySize);
	void NarrowPhase(glm::vec3* d_oldPositions, glm::vec3* d_positions, int arraySize);
	void NrCollisions(int arraySize);

private:
	void Init(int size);

	unsigned int* d_mortonKeys;
	unsigned int* d_mortonKeysAux;
	unsigned int* d_mortonValues;
	unsigned int* d_mortonValuesAux;
	unsigned int* d_keysColliding;
	int* d_atom;
	Node* d_leafNodes;
	Node* d_internalNodes;
	glm::vec3* d_finalPositions;
};