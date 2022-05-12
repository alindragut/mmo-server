#include "SimulationSettings.cuh"

float SimulationSettings::m_speed = 0.0f;
unsigned int SimulationSettings::m_tickRate = 0;
unsigned int SimulationSettings::m_maxNumberOfPlayers = 0;
glm::ivec2 SimulationSettings::m_heightmapDims = glm::ivec2(0);
float SimulationSettings::m_heightmapMaxHeight = 0.0f;
std::string SimulationSettings::m_heightmapPath = std::string();
