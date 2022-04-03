#include "Serialization.cuh"

#include <string.h>
#include <iostream>


Serialization::PlayerSnapshotPacket::PlayerSnapshotPacket(unsigned char* data) {
	memcpy(&m_playerID, &data[0], sizeof(unsigned short));
	memcpy(&m_pos.x, &data[2], sizeof(float));
	memcpy(&m_pos.y, &data[6], sizeof(float));
	memcpy(&m_pos.z, &data[10], sizeof(float));

	// std::cout << "Player position packet: " << m_playerID << " " << m_pos.x << " " << m_pos.y << " " << m_pos.z << std::endl;
}

void Serialization::PlayerSnapshotPacket::Encode(unsigned char* data, const glm::vec3& m_pos, unsigned short m_playerID) {
	memcpy(&data[0], &m_playerID, sizeof(unsigned short));
	memcpy(&data[2], &m_pos.x, sizeof(float));
	memcpy(&data[6], &m_pos.y, sizeof(float));
	memcpy(&data[10], &m_pos.z, sizeof(float));
}

Serialization::PlayerActionPacket::PlayerActionPacket(unsigned char* data)
{
	memcpy(&m_playerID, &data[0], sizeof(short));
	memcpy(&m_action, &data[2], sizeof(unsigned char));

	//std::cout << "Player action packet: " << m_playerID << " " << int(m_action) << std::endl;
}

void Serialization::Parse(unsigned char* data, int len, GameState& gameState) {
	switch (len) {
	case ACTIONPACKET: {
		PlayerActionPacket actionPacket(data);
		gameState.AddActionToEntityId(actionPacket.m_playerID, actionPacket.m_action);
		break;
	}
	case READYPACKET: {
		if (!strcmp((char*)data, "Ready")) {
			gameState.SetReady();
			std::cout << "Ready packet" << std::endl;
		}
		break;
	}
	case SNAPSHOTPACKET: {
		PlayerSnapshotPacket snapshotPacket(data);
		gameState.AddEntity(snapshotPacket.m_pos);
		break;
	}
	default:
		std::cout << "Default parse on packet: " << data << " len: " << len << std::endl;
		break;
	}
}