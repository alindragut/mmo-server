#pragma once

#include "CommonTypes.cuh"
#include "GameState.cuh"

#define SNAPSHOTPACKET 0
#define ACTIONPACKET 1
#define READYPACKET 2

namespace Serialization
{
	typedef struct PlayerSnapshotPacket
	{
		glm::vec3 m_pos;
		unsigned short m_playerID;

		PlayerSnapshotPacket(unsigned short playerID, const glm::vec3& pos) :
			m_playerID(playerID),
			m_pos(pos)
		{}

		PlayerSnapshotPacket(unsigned char* data, GameState& gameState);

		static void Encode(unsigned char* data, const glm::vec3& m_pos, unsigned short m_playerID);

	} PlayerSnapshotPacket;

	typedef struct PlayerActionPacket
	{
		unsigned short m_playerID;
		unsigned char m_action;

		PlayerActionPacket(unsigned short playerID, unsigned char action) :
			m_playerID(playerID),
			m_action(action)
		{}

		PlayerActionPacket(unsigned char* data, GameState& gameState);

	} PlayerActionPacket;

	void Parse(unsigned char* data, int len, GameState& gameState);
}