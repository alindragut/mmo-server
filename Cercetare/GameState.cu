#include "GameState.cuh"
#include "SimulationSettings.cuh"
#include <iostream>

__global__ void updatePositions(unsigned char* actions, glm::vec3* positions, float speed, int N)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < N) {
		positions[idx].y -= speed;
		switch (actions[idx]) {
			case 0:
				break;
			case 1:
				positions[idx].z += speed;
				break;
			case 2:
				positions[idx].x += speed;
				break;
			case 3:
				positions[idx].z -= speed;
				break;
			case 4:
				positions[idx].x -= speed;
				break;
			default:
				break;
		}
	}
}

GameState::GameState() {
	int N = SimulationSettings::GetMaxNumberOfPlayers();

	m_initOldPositions = true;
	m_ready = false;
	m_nrEntities = 0;

	CubDebugExit(cudaMallocHost((void**)&h_positions, sizeof(glm::vec3) * N));
	CubDebugExit(cudaMallocHost((void**)&h_actions, sizeof(unsigned char) * N));

	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_positions, sizeof(glm::vec3) * N));
	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_oldPositions, sizeof(glm::vec3) * N));
	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_actions, sizeof(unsigned char) * N));
}

GameState::~GameState() {
	CubDebugExit(cudaFreeHost(h_positions));
	CubDebugExit(cudaFreeHost(h_actions));
}

void GameState::AddActionToEntityId(unsigned int id, unsigned char action) {
	if (id > m_nrEntities) {
		return;
	}

	h_actions[id] = action;
}

void GameState::ApplyForces() {
	CubDebugExit(cudaMemcpy(d_positions, h_positions, sizeof(glm::vec3) * m_nrEntities, cudaMemcpyHostToDevice));
	CubDebugExit(cudaMemcpy(d_actions, h_actions, sizeof(unsigned char) * m_nrEntities, cudaMemcpyHostToDevice));

	int blockSize = 256;
	int numBlocks = (m_nrEntities + blockSize - 1) / blockSize;

	updatePositions<<<numBlocks, blockSize>>>(d_actions, d_positions, SimulationSettings::GetSpeed(), m_nrEntities);

	if (m_initOldPositions) {
		m_initOldPositions = false;
		CubDebugExit(cudaMemcpy(d_oldPositions, d_positions, sizeof(glm::vec3) * m_nrEntities, cudaMemcpyDeviceToDevice));
	}

	// memset(h_actions, 0, m_nrEntities);
}

void GameState::UpdateHostPositions() {
	CubDebugExit(cudaMemcpy(h_positions, d_positions, sizeof(glm::vec3) * m_nrEntities, cudaMemcpyDeviceToHost));
}