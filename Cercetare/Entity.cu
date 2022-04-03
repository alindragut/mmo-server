#include "Entity.cuh"
#include "SimulationSettings.cuh"

Entity::Entity(unsigned short id, const glm::vec3& position) :
	m_id(id),
	m_position(position)
{}

Entity::~Entity() 
{}

bool Entity::AddAction(unsigned char action)
{
	if (m_actions.size() > m_maxActions) {
		return false;
	}

	m_actions.push_back(action);

	return true;
}

void Entity::ApplyActions()
{
	for (unsigned char action : m_actions) {
		ApplyAction(action);
	}

	m_actions.clear();
}

void Entity::ApplyAction(unsigned char action)
{
	float offset = SimulationSettings::GetSpeed();

	switch (action) {
		case 0:
			break;
		case 1:
			m_position.z += offset;
			break;
		case 2:
			m_position.x += offset;
			break;
		case 3:
			m_position.z -= offset;
			break;
		case 4:
			m_position.x -= offset;
			break;
		default:
			break;
	}
}