// Cercetare.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "Server.cuh"
#include "SimulationSettings.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include "jdbc/mysql_driver.h"
#include "jdbc/mysql_connection.h"
#include "jdbc/cppconn/driver.h"
#include "jdbc/cppconn/exception.h"
#include "jdbc/cppconn/resultset.h"
#include "jdbc/cppconn/statement.h"
#include "World.cuh"

int main()
{
	SimulationSettings::SetTickRate(60);
	SimulationSettings::SetSpeed(5.0f);
	
	SimulationSettings::SetMaxNumberOfPlayers(1 << 16);

	SimulationSettings::SetHeightmapDims(glm::ivec2(1024, 1024));
	SimulationSettings::SetHeightmapMaxHeight(160.0f);
	SimulationSettings::SetHeightmapPath(std::string("E:\\Facultate\\Facultate\\Cercetare\\cercetare\\terrain_new.raw"));

	/*try {
		sql::Driver* driver;
		sql::Connection* con;
		sql::Statement* stmt;
		sql::ResultSet* res;

		driver = get_driver_instance();
		con = driver->connect("tcp://127.0.0.1:3306", "root", "doom2iddqd");
		con->setSchema("mmo");

		stmt = con->createStatement();
		res = stmt->executeQuery("SELECT 'Hello World!' AS _message");
		while (res->next()) {
			std::cout << "\t... MySQL replies: ";
			std::cout << res->getString("_message") << std::endl;
			std::cout << "\t... MySQL says it again: ";
			std::cout << res->getString(1) << std::endl;
		}
		delete res;
		delete stmt;
		delete con;

	}
	catch (sql::SQLException& e) {
		std::cout << "# ERR: SQLException in " << __FILE__;
		std::cout << "(" << __FUNCTION__ << ") on line " << __LINE__ << std::endl;
		std::cout << "# ERR: " << e.what();
		std::cout << " (MySQL error code: " << e.getErrorCode();
		std::cout << ", SQLState: " << e.getSQLState() << " )" << std::endl;
	}*/
	Server server(30000);

	server.Start();

	return 0;
}
