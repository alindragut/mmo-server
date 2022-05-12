#pragma once

#include "CommonTypes.cuh"

class SimulationSettings
{
public:
	static void SetSpeed(float speed) { m_speed = speed; }
	static float GetSpeed() { return m_speed; }
	static void SetTickRate(unsigned int tickRate) { m_tickRate = tickRate; }
	static unsigned int GetTickRate() { return m_tickRate; }
	static void SetMaxNumberOfPlayers(unsigned int maxPlayers) { m_maxNumberOfPlayers = maxPlayers; }
	static unsigned int GetMaxNumberOfPlayers() { return m_maxNumberOfPlayers; }
	static void SetHeightmapDims(const glm::ivec2& dims) { m_heightmapDims = dims; }
	static glm::ivec2& GetHeightmapDims() { return m_heightmapDims; }
	static void SetHeightmapMaxHeight(float maxHeight) { m_heightmapMaxHeight = maxHeight; }
	static float GetHeightmapMaxHeight() { return m_heightmapMaxHeight; }
	static void SetHeightmapPath(const std::string& path) { m_heightmapPath = path; }
	static std::string& GetHeightmapPath() { return m_heightmapPath; }

private:
	static float m_speed;
	static unsigned int m_tickRate;
	static unsigned int m_maxNumberOfPlayers;
	static glm::ivec2 m_heightmapDims;
	static float m_heightmapMaxHeight;
	static std::string m_heightmapPath;
};