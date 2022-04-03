#pragma once
#include "CommonTypes.cuh"

namespace Kernels {
	void TestKernel();
	void CollisionCheck(glm::vec3** oldPositions, glm::vec3** currentPositions, glm::vec3** finalPositions, int arraySize);
	//void CollisionCheckGPU(glm::vec3* oldPositions, glm::vec3* currentPositions, glm::vec3* finalPositions, int arraySize);
	void CollisionCheckCPU(glm::vec3* oldPositions, glm::vec3* currentPositions, glm::vec3* finalPositions, int arraySize);
}