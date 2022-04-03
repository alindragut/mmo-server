#pragma once
#include "GameState.cuh"
#include "Bvh.cuh"

class Physics
{
public:
	Physics();
	~Physics();

	void Step(GameState& gameState, bool verbose);

private:
	BVH m_bvh;
};