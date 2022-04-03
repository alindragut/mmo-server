#pragma once

#include "CommonTypes.cuh"
#include <vector>

class Entity
{
public:
	Entity(unsigned short id, const glm::vec3& position);
	~Entity();

	const glm::vec3& GetPosition() { return m_position; }
	void SetPosition(const glm::vec3& position) { m_position = position; }

	unsigned short GetId() { return m_id; }


	bool AddAction(unsigned char action);

	void ApplyActions();

private:
	void ApplyAction(unsigned char action);

	const unsigned short m_maxActions = 1;
	unsigned short m_id;
	std::vector<unsigned char> m_actions;
	glm::vec3 m_position;
};