#pragma once

class SimulationSettings
{
public:
	static void SetSpeed(float speed) { m_speed = speed; }
	static float GetSpeed() { return m_speed; }
	static void SetTickRate(unsigned int tickRate) { m_tickRate = tickRate; }
	static unsigned int GetTickRate() { return m_tickRate; }
	static void SetMaxNumberOfPlayers(unsigned int maxPlayers) { m_maxNumberOfPlayers = maxPlayers; }
	static unsigned int GetMaxNumberOfPlayers() { return m_maxNumberOfPlayers; }

private:
	static float m_speed;
	static unsigned int m_tickRate;
	static unsigned int m_maxNumberOfPlayers;
};