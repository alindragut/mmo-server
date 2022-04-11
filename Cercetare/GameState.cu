#include "GameState.cuh"
#include "SimulationSettings.cuh"
#include <iostream>

__global__ void updatePositions(unsigned char* actions, glm::vec3* oldPositions, glm::vec3* impulses, glm::vec3* corrections, glm::vec3* positions, float* collisionsNr, float speed, int N)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < N) {
		glm::vec3& pos = positions[idx];
		glm::vec3& oldPos = oldPositions[idx];
		glm::vec3 forces = glm::vec3(0.0f, -10.0f, 0.0f);
		glm::vec3 velocity = pos - oldPos + impulses[idx]; // / collisionsNr[idx]; // poate impulses derivat in functie de oldPOs ca sa nu mai fie alocat si impulses
		
		oldPos = pos;

		float deltaTimeSq = 1.0f / 3600.0f;

		pos += velocity * 0.98f + forces * deltaTimeSq; // <<<< tangential impulse s-ar putea sa fie nevoie

		pos += corrections[idx];
		/*switch (actions[idx]) {
			case 0:
				break;
			case 1:
				pos.z += speed;
				break;
			case 2:
				pos.x += speed;
				break;
			case 3:
				pos.z -= speed;
				break;
			case 4:
				pos.x -= speed;
				break;
			default:
				break;
		}*/


		pos.x = fminf(fmaxf(pos.x, 2.0f), 1023.0f);
		pos.y = fminf(fmaxf(pos.y, 2.0f), 1023.0f);
		pos.z = fminf(fmaxf(pos.z, 2.0f), 1023.0f);
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
	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_impulses, sizeof(glm::vec3) * N));
	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_corrections, sizeof(glm::vec3) * N));
	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_actions, sizeof(unsigned char) * N));
	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_collisionsNr, sizeof(float) * N));
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
	int blockSize = 256;
	int numBlocks = (m_nrEntities + blockSize - 1) / blockSize;

	updatePositions<<<numBlocks, blockSize>>>(d_actions, d_oldPositions, d_impulses, d_corrections, d_positions, d_collisionsNr, SimulationSettings::GetSpeed(), m_nrEntities);

	UpdateHost();
	// memset(h_actions, 0, m_nrEntities);
}

void GameState::UpdateDevice() {
	CubDebugExit(cudaMemcpy(d_positions, h_positions, sizeof(glm::vec3) * m_nrEntities, cudaMemcpyHostToDevice));
	CubDebugExit(cudaMemcpy(d_actions, h_actions, sizeof(unsigned char) * m_nrEntities, cudaMemcpyHostToDevice));
	CubDebugExit(cudaMemset(d_impulses, 0, sizeof(glm::vec3) * m_nrEntities));
	CubDebugExit(cudaMemset(d_corrections, 0, sizeof(glm::vec3) * m_nrEntities));
	CubDebugExit(cudaMemset(d_collisionsNr, 0, sizeof(float) * m_nrEntities));

	if (m_initOldPositions) {
		m_initOldPositions = false;
		CubDebugExit(cudaMemcpy(d_oldPositions, d_positions, sizeof(glm::vec3) * m_nrEntities, cudaMemcpyDeviceToDevice));
	}
}

void GameState::UpdateHost() {
	CubDebugExit(cudaMemcpy(h_positions, d_positions, sizeof(glm::vec3) * m_nrEntities, cudaMemcpyDeviceToHost));
}