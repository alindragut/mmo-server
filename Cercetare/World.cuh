#pragma once

#include "CommonTypes.cuh"

class World {
public:
	World();
	~World();

	void LoadHeightmap(const std::string& path, const glm::ivec2& dims, float maxHeight);

	glm::vec2 GetMaxDimensions() { return m_dims; }

	void CheckBoundaries(Shape* d_shapes, glm::vec3* d_positions, int arraySize);

private:
	cudaTextureObject_t d_tex;
	cudaArray_t d_array;
	glm::vec2 m_dims;
};
