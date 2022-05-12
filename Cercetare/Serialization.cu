#include "Serialization.cuh"

#include <string.h>
#include <iostream>


Serialization::PlayerSnapshotPacket::PlayerSnapshotPacket(unsigned char* data, GameState& gameState) {
	memcpy(&m_playerID, &data[1], sizeof(unsigned short));
	memcpy(&m_pos.x, &data[3], sizeof(float));
	memcpy(&m_pos.y, &data[7], sizeof(float));
	memcpy(&m_pos.z, &data[11], sizeof(float));

	std::cout << "Player position packet: " << m_playerID << " " << m_pos.x << " " << m_pos.y << " " << m_pos.z << std::endl;

	ShapeType type = (ShapeType)(1 + m_playerID % 3);
	gameState.AddEntity(m_pos, type);
}

void Serialization::PlayerSnapshotPacket::Encode(unsigned char* data, const glm::vec3& m_pos, unsigned short m_playerID) {
	memcpy(&data[0], &m_playerID, sizeof(unsigned short));
	memcpy(&data[2], &m_pos.x, sizeof(float));
	memcpy(&data[6], &m_pos.y, sizeof(float));
	memcpy(&data[10], &m_pos.z, sizeof(float));
}

Serialization::PlayerActionPacket::PlayerActionPacket(unsigned char* data, GameState& gameState)
{
	unsigned char actionsSize = data[1];

	unsigned char* dataPtr = data + 2;

	for (unsigned char i = 0; i < actionsSize; i++, dataPtr += 3) {
		memcpy(&m_playerID, dataPtr, sizeof(short));
		memcpy(&m_action, dataPtr + 2, sizeof(unsigned char));
		gameState.AddActionToEntityId(m_playerID, m_action);
	}
}

void Serialization::Parse(unsigned char* data, int len, GameState& gameState) {
	unsigned char packetType = data[0];

	switch (packetType) {
	case SNAPSHOTPACKET: {
		PlayerSnapshotPacket snapshotPacket(data, gameState);
		break;
	}
	case ACTIONPACKET: {
		PlayerActionPacket actionPacket(data, gameState);
		break;
	}
	case READYPACKET: {
		gameState.SetReady();
		break;
	}
	default:
		std::cout << "Default parse on packet: " << data << " len: " << len << std::endl;
		break;
	}
}