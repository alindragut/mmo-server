#include "Physics.cuh"
#include "SimulationSettings.cuh"
#include "Kernels.cuh"
#include <iostream>
#include <chrono>

Physics::Physics() {}

Physics::~Physics() {}

void Physics::Step(GameState& gameState, bool verbose)
{
	float time;
	cudaEvent_t start, stop;

	glm::vec3* d_positions = gameState.GetDevicePositions();
	unsigned int nrEntities = gameState.GetNrEntities();

	if (verbose) {
		printf("Starting physics with %d entities\n", nrEntities);
	}

	CubDebugExit(cudaEventCreate(&start));
	CubDebugExit(cudaEventCreate(&stop));

	CubDebugExit(cudaEventRecord(start, 0));

	gameState.UpdateDevice();

	CubDebugExit(cudaEventRecord(stop, 0));
	CubDebugExit(cudaEventSynchronize(stop));
	CubDebugExit(cudaEventElapsedTime(&time, start, stop));

	if (verbose) {
		printf("Time to update device buffers:  %3.3f ms \n", time);
	}

	CubDebugExit(cudaEventRecord(start, 0));

	m_bvh.Build(d_positions, nrEntities);

	CubDebugExit(cudaEventRecord(stop, 0));
	CubDebugExit(cudaEventSynchronize(stop));
	CubDebugExit(cudaEventElapsedTime(&time, start, stop));

	if (verbose) {
		printf("Time to build BVH:  %3.3f ms \n", time);
	}

	CubDebugExit(cudaEventRecord(start, 0));

	m_bvh.BroadPhase(d_positions, nrEntities);

	CubDebugExit(cudaEventRecord(stop, 0));
	CubDebugExit(cudaEventSynchronize(stop));
	CubDebugExit(cudaEventElapsedTime(&time, start, stop));

	if (verbose) {
		printf("Time for physics broad phase:  %3.3f ms \n", time);
	}

	CubDebugExit(cudaEventRecord(start, 0));

	m_bvh.NarrowPhase(gameState.GetDeviceOldPositions(), d_positions, gameState.GetDeviceImpulses(), gameState.GetDeviceCorrections(), gameState.GetDeviceCollisionsNr(), nrEntities);

	CubDebugExit(cudaEventRecord(stop, 0));
	CubDebugExit(cudaEventSynchronize(stop));
	CubDebugExit(cudaEventElapsedTime(&time, start, stop));

	if (verbose) {
		printf("Time for physics narrow phase:  %3.3f ms \n", time);
	}

	CubDebugExit(cudaEventRecord(start, 0));

	gameState.ApplyForces();

	CubDebugExit(cudaEventRecord(stop, 0));
	CubDebugExit(cudaEventSynchronize(stop));
	CubDebugExit(cudaEventElapsedTime(&time, start, stop));

	if (verbose) {
		printf("Time to apply forces:  %3.3f ms \n", time);
	}

	// m_bvh.NrCollisions(nrEntities);
}


