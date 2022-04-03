#include "Address.cuh"

Address::Address() :
    m_address(0),
    m_port(0)
{}

Address::Address(unsigned char a,
    unsigned char b,
    unsigned char c,
    unsigned char d,
    unsigned short port)
{
    m_address = (a << 24) |
        (b << 16) |
        (c << 8) |
        d;
    m_port = port;
}

Address::Address(unsigned int address,
    unsigned short port) :
    m_address(address),
    m_port(port)
{}

unsigned int Address::GetAddress() const {
    return m_address;
}

unsigned char Address::GetA() const {
    return m_address >> 24;
}

unsigned char Address::GetB() const {
    return (m_address << 8) >> 24;
}

unsigned char Address::GetC() const {
    return (m_address << 16) >> 24;
}

unsigned char Address::GetD() const {
    return (m_address << 24) >> 24;
}

unsigned short Address::GetPort() const {
    return m_port;
}