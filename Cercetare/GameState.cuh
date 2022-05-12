#pragma once

#include "CommonTypes.cuh"
#include "World.cuh"

class GameState
{
public:
	GameState();
	~GameState();

	void SetReady() { m_ready = true; }
	bool IsReady() { return m_ready; }

	void AddEntity(glm::vec3 pos, ShapeType type) { 
		switch (type) {
		case ShapeType::SphereShape:
			h_shapes[m_nrEntities].sphere.radius = 1.0f;
			break;
		case ShapeType::BoxShape:
			h_shapes[m_nrEntities].box.halfExtents = glm::vec3(1.0f);
			break;
		case ShapeType::CapsuleShape:
			h_shapes[m_nrEntities].capsule.halfHeight = 1.5f;
			h_shapes[m_nrEntities].capsule.radius = 1.0f;
			break;
		default:
			return;
		}

		h_shapes[m_nrEntities].type = type;
		h_positions[m_nrEntities++] = pos;
	}
	void AddActionToEntityId(unsigned int id, unsigned char action);

	void UpdateDevice();
	void ApplyForces();

	glm::vec3* GetDevicePositions() { return d_positions; }
	glm::vec3* GetDeviceOldPositions() { return d_oldPositions; }
	glm::vec3* GetDeviceImpulses() { return d_impulses; }
	glm::vec3* GetDeviceCorrections() { return d_corrections; }
	float* GetDeviceCollisionsNr() { return d_collisionsNr; }
	glm::vec3* GetHostPositions() { return h_positions; }
	unsigned int GetNrEntities() { return m_nrEntities; }
	Shape* GetDeviceShapes() { return d_shapes; }

private:
	void UpdateHost();

	World m_world;
	bool m_ready;
	bool m_initOldPositions;
	unsigned int m_nrEntities;
	unsigned char* h_actions;
	glm::vec3* h_positions;
	unsigned char* d_actions;
	glm::vec3* d_positions;
	glm::vec3* d_oldPositions;
	glm::vec3* d_impulses;
	glm::vec3* d_corrections;
	float* d_collisionsNr;
	Shape* h_shapes;
	Shape* d_shapes;
};