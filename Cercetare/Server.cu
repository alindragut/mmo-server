#include "Server.cuh"
#include <iostream>
#include "Serialization.cuh"
#include "SimulationSettings.cuh"

Server::Server(int port) : m_port(port),
	m_socket(),
    m_gameState(),
    m_physics(),
    m_lastTimeStep(Clock::now()),
    m_msPerTick(1000 / SimulationSettings::GetTickRate()),
    m_snapshotBuf(NULL)
{
}

Server::~Server()
{
    if (m_snapshotBuf) {
        delete[] m_snapshotBuf;
    }
}

void Server::Start()
{
    Address receiver(127, 0, 0, 1, 30000);
    Address sender(192, 168, 0, 113, 30000);
    unsigned char buffer[512];
    int bytes;

    if (!m_socket.Open(m_port))
    {
        std::cout << "failed to create socket!\n";
        return;
    }

    int counter = 0;


    while (true)
    {
        memset(buffer, 0, sizeof(buffer));
        bytes = m_socket.Receive(sender, buffer, sizeof(buffer));
        
        if (bytes > 0) {
            Serialization::Parse(buffer, bytes, m_gameState);
        }

        if (RunPhysics() && m_gameState.IsReady()) {
            bool verbose = (counter++) % SimulationSettings::GetTickRate() == 0;
            if (verbose) {
                printf("Tick %d\n", counter - 1);
            }
            m_physics.Step(m_gameState, verbose);
            time_point<Clock> beforeSending = Clock::now();
            SendSnapshot(sender);
            time_point<Clock> afterSending = Clock::now();
            int diff = duration_cast<microseconds>(afterSending - beforeSending).count();

            if (verbose) {
                printf("Time for sending packets: %3.3f ms\n\n", diff / 1000.0f);
            }
        }
    }
}

bool Server::RunPhysics()
{
    time_point<Clock> now = Clock::now();
    int diff = duration_cast<milliseconds>(now - m_lastTimeStep).count();

    if (diff >= m_msPerTick) {
        m_lastTimeStep = now;
        return true;
    }

    return false;
}

#define SNAPSHOT_PACKET_SIZE 14
#define ENTITIES_PER_PACKET 30
#define MAX_PACKET_SIZE SNAPSHOT_PACKET_SIZE * ENTITIES_PER_PACKET
void Server::SendSnapshot(const Address& address)
{
    if (!m_snapshotBuf) {
        m_snapshotBuf = new unsigned char[(SNAPSHOT_PACKET_SIZE + 1) * m_gameState.GetNrEntities()];
    }

    unsigned int nrEntities = m_gameState.GetNrEntities();
    glm::vec3* positions = m_gameState.GetHostPositions();

    
    int offset = 0;

    for (unsigned int i = 0; i < nrEntities; i++) {
        if (i % ENTITIES_PER_PACKET == 0) {
            m_snapshotBuf[offset++] = (nrEntities - i) > ENTITIES_PER_PACKET ? ENTITIES_PER_PACKET : (nrEntities - i);
        }

        memcpy(m_snapshotBuf + offset, &i, sizeof(short));
        memcpy(m_snapshotBuf + offset + 2, positions + i, sizeof(glm::vec3));

        offset += SNAPSHOT_PACKET_SIZE;
    }

    int bufOffset = 0;

    while (offset != 0) {
        int size = MAX_PACKET_SIZE + 1;

        if (offset < MAX_PACKET_SIZE + 1) {
            size = offset;
        }

        // printf("size: %d offset: %d\n", size, bufOffset);

        offset = std::max(0, offset - MAX_PACKET_SIZE - 1);

        m_socket.Send(address, m_snapshotBuf + bufOffset, size);

        bufOffset += size;
    }
}