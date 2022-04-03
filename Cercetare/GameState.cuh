#pragma once

#include "CommonTypes.cuh"

class GameState
{
public:
	GameState();
	~GameState();

	void SetReady() { m_ready = true; }
	bool IsReady() { return m_ready; }

	void AddEntity(glm::vec3 pos) { h_positions[m_nrEntities++] = pos; }
	void AddActionToEntityId(unsigned int id, unsigned char action);

	void ApplyForces();
	void UpdateHostPositions();

	glm::vec3* GetDevicePositions() { return d_positions; }
	glm::vec3* GetDeviceOldPositions() { return d_oldPositions; }
	glm::vec3* GetHostPositions() { return h_positions; }
	unsigned int GetNrEntities() { return m_nrEntities; }

private:
	bool m_ready;
	bool m_initOldPositions;
	unsigned int m_nrEntities;
	unsigned char* h_actions;
	glm::vec3* h_positions;
	unsigned char* d_actions;
	glm::vec3* d_positions;
	glm::vec3* d_oldPositions;
};