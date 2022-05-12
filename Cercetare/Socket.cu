#include "Socket.cuh"
#include <winsock2.h>
#include <iostream>

bool Socket::InitializeSockets()
{
    WSADATA WsaData;
    return WSAStartup(MAKEWORD(2, 2),
        &WsaData)
        == NO_ERROR;
}

void Socket::ShutdownSockets()
{
    WSACleanup();
}

Socket::Socket() : m_handle(-1)
{
    if (!InitializeSockets()) {
        std::cout << "Failed to initialize sockets";
    }
}

Socket::~Socket() {
    Close();
    ShutdownSockets();
}

bool Socket::Open(unsigned short port) {
    m_handle = socket(AF_INET,
        SOCK_DGRAM,
        IPPROTO_UDP);

    if (m_handle <= 0)
    {
        printf("failed to create socket\n");
        return false;
    }

    sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port =
        htons((unsigned short)port);

    if (bind(m_handle,
        (const sockaddr*)&address,
        sizeof(sockaddr_in)) < 0)
    {
        printf("failed to bind socket\n");
        return false;
    }

    DWORD nonBlocking = 1;
    if (ioctlsocket(m_handle,
        FIONBIO,
        &nonBlocking) != 0)
    {
        printf("failed to set non-blocking\n");
        return false;
    }

    return true;
}

void Socket::Close() {
    closesocket(m_handle);
    m_handle = -1;
}

bool Socket::IsOpen() const {
    return m_handle > 0;
}

bool Socket::Send(const Address& destination,
    const void* data,
    int size) {
    sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(destination.GetAddress());
    addr.sin_port = htons(destination.GetPort());

    // printf("Sending to addr: %d %d %d %d port %d\n", destination.GetA(), destination.GetB(), destination.GetC(), destination.GetD(), destination.GetPort());

    int sent_bytes =
        sendto(m_handle,
            (const char*)data,
            size,
            0,
            (sockaddr*)&addr,
            sizeof(sockaddr_in));

    if (sent_bytes != size)
    {
        // printf("failed to send packet\n");
        return false;
    }

    return true;
}

int Socket::Receive(Address& sender,
    void* data,
    int size) {

    typedef int socklen_t;

    sockaddr_in from;
    socklen_t fromLength = sizeof(from);

    int bytes = recvfrom(m_handle,
        (char*)data,
        size,
        0,
        (sockaddr*)&from,
        &fromLength);

    if (bytes <= 0)
        return 0;

    /*unsigned int from_address =
        ntohl(from.sin_addr.s_addr);

    unsigned int from_port =
        ntohs(from.sin_port);

    Address fromAddress(from_address, from_port);

    std::cout << "Packet from " << int(fromAddress.GetA()) << "." <<
        int(fromAddress.GetB()) << "." <<
        int(fromAddress.GetC()) << "." <<
        int(fromAddress.GetD()) << " " << from_port << " " << data << std::endl;*/
        
    return bytes;
}