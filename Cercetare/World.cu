#include "World.cuh"
#include <iostream>
#include <fstream>
#include <vector>

__global__ void checkBoundaries(glm::vec3* positions, cudaTextureObject_t texObj, glm::vec2 dims, int N)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < N) {
		glm::vec3& pos = positions[idx];
		pos.x = fminf(fmaxf(pos.x, 2.0f), dims.x);
		pos.z = fminf(fmaxf(pos.z, 2.0f), dims.y);

		float u = (pos.x + 0.5f) / dims.x;
		float v = (pos.z + 0.5f) / dims.y;

		pos.y = tex2D<float>(texObj, u, v);
	}
}

World::World() {}

World::~World() {
	CubDebugExit(cudaDestroyTextureObject(d_tex));
	CubDebugExit(cudaFreeArray(d_array));
}

void World::LoadHeightmap(const std::string& path, const glm::ivec2& dims, float maxHeight) {
	std::ifstream is(path, std::ios::binary);

	unsigned short* heightmap = new unsigned short[dims.x * dims.y];
    float* heightmapFloat = new float[dims.x * dims.y];

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	CubDebugExit(cudaMallocArray(&d_array, &channelDesc, dims.x, dims.y));

	is.read((char*)heightmap, dims.x * dims.y * sizeof(unsigned short));

	int max = -1;

	for (unsigned int i = 0; i < dims.x * dims.y; i++) {
		heightmapFloat[i] = ((float)(heightmap[i]) / USHRT_MAX) * maxHeight;
	}

	m_dims = dims;

	const size_t spitch = dims.x * sizeof(float);

	CubDebugExit(cudaMemcpy2DToArray(d_array, 0, 0, heightmapFloat, spitch, sizeof(float) * dims.x, dims.y, cudaMemcpyHostToDevice));

	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = d_array;

	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 1;

	d_tex = 0;
	CubDebugExit(cudaCreateTextureObject(&d_tex, &resDesc, &texDesc, NULL));

	delete[] heightmapFloat;
	delete[] heightmap;

	is.close();
}

void World::CheckBoundaries(Shape* d_shapes, glm::vec3* d_positions, int arraySize) {
	int blockSize = 256;
	int numBlocks = (arraySize + blockSize - 1) / blockSize;

	checkBoundaries<<<numBlocks, blockSize>>>(d_positions, d_tex, m_dims, arraySize);

	CubDebugExit(cudaDeviceSynchronize());
}
