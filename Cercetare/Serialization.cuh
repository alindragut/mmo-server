#pragma once

#include "CommonTypes.cuh"
#include "GameState.cuh"

#define ACTIONPACKET 3
#define READYPACKET 5
#define SNAPSHOTPACKET 14

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

		PlayerSnapshotPacket(unsigned char* data);

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

		PlayerActionPacket(unsigned char* data);

	} PlayerActionPacket;

	void Parse(unsigned char* data, int len, GameState& gameState);
}