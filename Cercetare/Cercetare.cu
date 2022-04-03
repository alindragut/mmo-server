// Cercetare.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "Server.cuh"
#include "SimulationSettings.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <vector>

int main()
{
	SimulationSettings::SetTickRate(60);
	SimulationSettings::SetSpeed(0.05f);
	
	SimulationSettings::SetMaxNumberOfPlayers(1 << 16);

	Server server(30000);

	server.Start();

	return 0;
}
