#pragma once

#include "Socket.cuh"
#include "Physics.cuh"
#include "Entity.cuh"
#include <chrono>

using Clock = std::chrono::steady_clock;
using std::chrono::time_point;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::microseconds;

class Server
{
public:
	Server(int port);
	~Server();

	void Start();

	bool RunPhysics();

	void SendSnapshot(const Address& address);

private:
	int m_port;
	int m_msPerTick;
	unsigned char* m_snapshotBuf;
	Socket m_socket;
	GameState m_gameState;
	Physics m_physics;
	time_point<Clock> m_lastTimeStep;
};