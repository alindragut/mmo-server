#include "Bvh.cuh"
#include "SimulationSettings.cuh"

#define MAX_COLLISIONS 26

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__host__ __device__ unsigned int expandBits(unsigned int v)
{
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__host__ __device__ unsigned int morton3D(float x, float y, float z)
{
	unsigned int xx = expandBits((unsigned int)x);
	unsigned int yy = expandBits((unsigned int)y);
	unsigned int zz = expandBits((unsigned int)z);
	return xx * 4 + yy * 2 + zz;
}

__global__
void morton(int n, glm::vec3* centers, unsigned int* morton_codes, unsigned int* keys)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < n) {
		morton_codes[index] = morton3D(centers[index].x, centers[index].y, centers[index].z);
		keys[index] = index;
		//printf("Morton for %d - %d\n", index, morton_codes[index]);
	}
}

__device__ int delta(unsigned int* sortedMortonCodes, int x, int y, int numObjects)
{
	if (x >= 0 && x <= numObjects - 1 && y >= 0 && y <= numObjects - 1)
	{
		if (sortedMortonCodes[x] == sortedMortonCodes[y]) {
			unsigned long long int keyX = sortedMortonCodes[x];
			unsigned long long int keyY = sortedMortonCodes[y];
			keyX = keyX << 32 | x;
			keyY = keyY << 32 | y;
			return __clzll(keyX ^ keyY);
		}
		return __clz(sortedMortonCodes[x] ^ sortedMortonCodes[y]);
	}
	return -1;
} //fix - mai mari cred sferele

__device__ int findSplit(unsigned int* sortedMortonCodes, int first, int last)
{
	// Identical Morton codes => split the range in the middle.

	unsigned int firstCode = sortedMortonCodes[first];
	unsigned int lastCode = sortedMortonCodes[last];

	if (firstCode == lastCode)
		return (first + last) >> 1;

	// Calculate the number of highest bits that are the same
	// for all objects, using the count-leading-zeros intrinsic.

	int commonPrefix = __clz(firstCode ^ lastCode);

	// Use binary search to find where the next bit differs.
	// Specifically, we are looking for the highest object that
	// shares more than commonPrefix bits with the first one.

	int split = first; // initial guess
	int step = last - first;

	do
	{
		step = (step + 1) >> 1; // exponential decrease
		int newSplit = split + step; // proposed new position

		if (newSplit < last)
		{
			unsigned int splitCode = sortedMortonCodes[newSplit];
			int splitPrefix = __clz(firstCode ^ splitCode);

			if (splitPrefix > commonPrefix)
				split = newSplit; // accept proposal
		}
	} while (step > 1);

	return split;
}

__device__ int sign(int x)
{
	return (x > 0) - (x < 0);
}

__device__ int2 determineRange(unsigned int* sortedMortonCodes, int numObjects, int idx)
{
	int d = sign(delta(sortedMortonCodes, idx, idx + 1, numObjects) - delta(sortedMortonCodes, idx, idx - 1, numObjects));
	int dmin = delta(sortedMortonCodes, idx, idx - d, numObjects);
	int lmax = 2;
	while (delta(sortedMortonCodes, idx, idx + lmax * d, numObjects) > dmin)
		lmax = lmax * 2;
	int l = 0;
	for (int t = lmax >> 1; t >= 1; t >>= 1)
	{
		if (delta(sortedMortonCodes, idx, idx + (l + t) * d, numObjects) > dmin)
			l += t;
	}
	int j = idx + l * d;
	int2 range;
	range.x = min(idx, j);
	range.y = max(idx, j);
	return range;
}

#define RADIUS 1.0f
__global__ void initLeafNodes(int N, unsigned int* sortedObjectIDs, Node* leafNodes, glm::vec3* positions)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < N) {
		leafNodes[idx].idx = idx;
		leafNodes[idx].objectID = sortedObjectIDs[idx];
		leafNodes[idx].nodeType = NodeType::Leaf;
		glm::vec3& position = positions[sortedObjectIDs[idx]];
		glm::vec3 offset = glm::vec3(RADIUS);
		leafNodes[idx].aabb.min = position - offset;
		leafNodes[idx].aabb.max = position + offset;
	}
}

__global__ void createBVH(int SAMPLE_SIZE, unsigned int* sortedMortonCodes, unsigned int* sortedObjectIDs, Node* leafNodes, Node* internalNodes)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < SAMPLE_SIZE - 1)
	{
		int2 range = determineRange(sortedMortonCodes, SAMPLE_SIZE, idx);
		int first = range.x;
		int last = range.y;

		// Determine where to split the range.

		int split = findSplit(sortedMortonCodes, first, last);

		// Select childA.
		Node* childA;
		if (split == first)
		{
			childA = &leafNodes[split];
			// printf("%d <-- %d (%d %d %d)\n", split, idx, first, last, split);
		}
		else
		{
			childA = &internalNodes[split];
			childA->objectID = split;
			childA->nodeType = NodeType::Internal;
			// printf("%d <- %d (%d %d %d)\n", split, idx, first, last, split);
		}
		// Select childB.
		Node* childB;
		if (split + 1 == last)
		{
			childB = &leafNodes[split + 1];
			// printf("%d --> %d (%d %d %d)\n", idx, split + 1, first, last, split);
		}
		else
		{
			childB = &internalNodes[split + 1];
			childB->objectID = split + 1;
			childB->nodeType = NodeType::Internal;
			// printf("%d -> %d (%d %d %d)\n", idx, split + 1, first, last, split);
		}

		internalNodes[idx].leftNode = childA;
		internalNodes[idx].rightNode = childB;

		childA->parent = &internalNodes[idx];
		childB->parent = &internalNodes[idx];
	}
}

__global__ void computeInternalBBox(int N, int* atom, Node* leafNodes, Node* internalNodes)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < N)
	{
		Node* node = leafNodes[idx].parent;

		while (atomicCAS(&atom[node->objectID], 0, 1) == 1)
		{
			AABB& parentAABB = node->aabb;
			const glm::vec3& leftAABBmin = node->leftNode->aabb.min;
			const glm::vec3& leftAABBmax = node->leftNode->aabb.max;
			const glm::vec3& rightAABBmin = node->rightNode->aabb.min;
			const glm::vec3& rightAABBmax = node->rightNode->aabb.max;

			parentAABB.min.x = fminf(leftAABBmin.x, rightAABBmin.x);
			parentAABB.min.y = fminf(leftAABBmin.y, rightAABBmin.y);
			parentAABB.min.z = fminf(leftAABBmin.z, rightAABBmin.z);

			parentAABB.max.x = fmaxf(leftAABBmax.x, rightAABBmax.x);
			parentAABB.max.y = fmaxf(leftAABBmax.y, rightAABBmax.y);
			parentAABB.max.z = fmaxf(leftAABBmax.z, rightAABBmax.z);

			if (node->nodeType == NodeType::Root) return;

			node = node->parent;
		}
	}
}

__device__ __host__ bool checkOverlap(const AABB& a, const AABB& b) {
	return (a.min.x < b.max.x && a.max.x > b.min.x) &&
		(a.min.y < b.max.y && a.max.y > b.min.y) &&
		(a.min.z < b.max.z && a.max.z > b.min.z);
}

__device__ void traverseIterative(Node* node, AABB& bbox, unsigned int idx, glm::vec3* pos, unsigned int* keysColliding)
{
	// Allocate traversal stack from thread-local memory,
	// and push NULL to indicate that there are no postponed nodes.
	Node* stack[64];
	Node** stackPtr = stack;
	*stackPtr++ = nullptr; // push
	unsigned int keysCollidingIdx = idx * MAX_COLLISIONS;
	unsigned int nrCollisions = 0;

	do
	{
		// Check each child node for overlap.
		Node* childL = node->leftNode;
		Node* childR = node->rightNode;
		unsigned int childLId = childL->objectID;
		unsigned int childRId = childR->objectID;
		bool overlapL = checkOverlap(bbox, childL->aabb);
		bool overlapR = checkOverlap(bbox, childR->aabb);

		// Query overlaps a leaf node => report collision.
		if (overlapL && childL->nodeType == NodeType::Leaf && childLId > idx) {
			keysColliding[keysCollidingIdx + (nrCollisions++)] = childLId;
		}

		if (nrCollisions == MAX_COLLISIONS) {
			return;
		}

		if (overlapR && childL->nodeType == NodeType::Leaf && childRId > idx) {
			keysColliding[keysCollidingIdx + (nrCollisions++)] = childRId;
		}

		if (nrCollisions == MAX_COLLISIONS) {
			return;
		}

		// Query overlaps an internal node => traverse.
		bool traverseL = (overlapL && (childL->nodeType == NodeType::Internal));
		bool traverseR = (overlapR && (childR->nodeType == NodeType::Internal));

		if (!traverseL && !traverseR)
			node = *--stackPtr; // pop
		else
		{
			node = (traverseL) ? childL : childR;
			if (traverseL && traverseR)
				*stackPtr++ = childR; // push
		}
	} while (node != NULL);

	//printf("Node %d didn't hit anything\n", idx);
}

__global__ void queryBVH(Node* root, Node* nodes, int N, glm::vec3* pos, unsigned int* keysColliding) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < N) {
		traverseIterative(root, nodes[idx].aabb, nodes[idx].objectID, pos, keysColliding);
	}
}

#define penSlack 0.00f
#define linearProjPercent 0.8f

#define RADIUS_SUM (2.0f * RADIUS)
#define RADIUS_SUM_SQ RADIUS_SUM * RADIUS_SUM
__global__ void narrowPhase(unsigned int* keysColliding, glm::vec3* oldPos, glm::vec3* pos, glm::vec3* impulses, glm::vec3* corrections, float* d_collisionsNr, int N) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < N) {
		unsigned int otherCollisionIdx = keysColliding[idx];

		if (otherCollisionIdx == -1) {
			return;
		}

		unsigned int thisCollisionIdx = idx / MAX_COLLISIONS;

		glm::vec3 d = pos[otherCollisionIdx] - pos[thisCollisionIdx];

		float length_sq = glm::length2(d);

		if (length_sq < RADIUS_SUM_SQ && length_sq > 0.0f) {
			glm::vec3 relativeVelocity = d - oldPos[otherCollisionIdx] + oldPos[thisCollisionIdx];
			glm::vec3 collisionNormal = glm::normalize(d);

			float velNormalDot = glm::dot(relativeVelocity, collisionNormal);


			/*printf("Collision %d with %d velNormalDot %f\n\trelative velocity %f %f %f collision normal %f %f %f\n", thisCollisionIdx, otherCollisionIdx, velNormalDot,
				relativeVelocity.x, relativeVelocity.y, relativeVelocity.z,
				collisionNormal.x, collisionNormal.y, collisionNormal.z);
				*/
			if (velNormalDot > 0.0f) {
				return;
			}

			float depth = fmaxf(fabsf(sqrt(length_sq) - RADIUS_SUM) * 0.5f - penSlack, 0.0f);

			glm::vec3 correction = collisionNormal * depth * 0.5f;

			float impulseMagn = -velNormalDot * 0.5f;

			glm::vec3 impulse = collisionNormal * impulseMagn;
			
			glm::vec3& impulseA = impulses[thisCollisionIdx];
			glm::vec3& impulseB = impulses[otherCollisionIdx];

			atomicAdd(&impulseA.x, -impulse.x);
			atomicAdd(&impulseA.y, -impulse.y);
			atomicAdd(&impulseA.z, -impulse.z);

			atomicAdd(&impulseB.x, impulse.x);
			atomicAdd(&impulseB.y, impulse.y);
			atomicAdd(&impulseB.z, impulse.z);

			glm::vec3& correctionA = corrections[thisCollisionIdx];
			glm::vec3& correctionB = corrections[otherCollisionIdx];

			atomicAdd(&correctionA.x, -correction.x);
			atomicAdd(&correctionA.y, -correction.y);
			atomicAdd(&correctionA.z, -correction.z);

			atomicAdd(&correctionB.x, correction.x);
			atomicAdd(&correctionB.y, correction.y);
			atomicAdd(&correctionB.z, correction.z);

			atomicAdd(&d_collisionsNr[thisCollisionIdx], 1.0f);
			atomicAdd(&d_collisionsNr[otherCollisionIdx], 1.0f);

			glm::vec3 tangFriction = relativeVelocity - (collisionNormal * glm::dot(relativeVelocity, collisionNormal));

			if (fabsf(glm::length2(tangFriction)) <= FLT_EPSILON) {
				return;
			}

			tangFriction = glm::normalize(tangFriction);

			float frictionMagn = -glm::dot(relativeVelocity, tangFriction) * 0.5f;

			if (fabsf(frictionMagn) <= FLT_EPSILON) {
				return;
			}

			float friction = frictionMagn * 0.98f;

			if (frictionMagn > friction) {
				frictionMagn = impulseMagn * friction;
			}
			else if (frictionMagn < friction) {
				frictionMagn = -impulseMagn  * friction;
			}

			glm::vec3 tangImpulse = tangFriction * frictionMagn;

			// combinare cu impulse normal mai incolo ca sa fie doar un singur atomic add per componenta

			atomicAdd(&impulseA.x, -tangImpulse.x);
			atomicAdd(&impulseA.y, -tangImpulse.y);
			atomicAdd(&impulseA.z, -tangImpulse.z);

			atomicAdd(&impulseB.x, tangImpulse.x);
			atomicAdd(&impulseB.y, tangImpulse.y);
			atomicAdd(&impulseB.z, tangImpulse.z);
		}
	}
}

__global__ void printPaths(int N, Node* internalNodes) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx == 0) {
		Node* node = &internalNodes[idx];
		Node* leftNode = node->leftNode;
		Node* rightNode = node->rightNode;
		printf("%d type: %d bb: %f %f %f %f %f %f <- %d type: %d -> %d type: %d bb: %f %f %f %f %f %f\n",
			leftNode->objectID, leftNode->nodeType, leftNode->aabb.min.x, leftNode->aabb.min.y, leftNode->aabb.min.z, leftNode->aabb.max.x, leftNode->aabb.max.y, leftNode->aabb.max.z,
			node->objectID, node->nodeType,
			rightNode->objectID, rightNode->nodeType, rightNode->aabb.min.x, rightNode->aabb.min.y, rightNode->aabb.min.z, rightNode->aabb.max.x, rightNode->aabb.max.y, rightNode->aabb.max.z);
	}
}

BVH::BVH() {
	Init(SimulationSettings::GetMaxNumberOfPlayers());
}

BVH::~BVH() {}

void BVH::Init(int size) {
	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_mortonKeys, sizeof(unsigned int) * size));
	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_mortonKeysAux, sizeof(unsigned int) * size));
	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_mortonValues, sizeof(unsigned int) * size));
	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_mortonValuesAux , sizeof(unsigned int) * size));
	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_leafNodes, sizeof(Node) * size));
	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_internalNodes, sizeof(Node) * (size - 1)));
	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_atom, sizeof(int) * size));
	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keysColliding, sizeof(unsigned int) * size * MAX_COLLISIONS));
}

void BVH::Build(glm::vec3* d_positions, int arraySize) {
	int blockSize = 256;
	int numBlocks = (arraySize + blockSize - 1) / blockSize;

	morton<<<numBlocks, blockSize>>>(arraySize, d_positions, d_mortonKeys, d_mortonValues);

	void* d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;

	DoubleBuffer<unsigned int> d_keys(d_mortonKeys, d_mortonKeysAux);
	DoubleBuffer<unsigned int> d_values(d_mortonValues, d_mortonValuesAux);

	DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, arraySize);
	// Allocate temporary storage
	CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
	// Run sorting operation
	DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, arraySize);

	CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

	initLeafNodes<<<numBlocks, blockSize>>>(arraySize, d_values.Current(), d_leafNodes, d_positions);

	createBVH<<<numBlocks, blockSize>>>(arraySize, d_keys.Current(), d_values.Current(), d_leafNodes, d_internalNodes);

	CubDebugExit(cudaMemset(d_atom, 0, sizeof(int) * arraySize));

	computeInternalBBox<<<numBlocks, blockSize>>>(arraySize, d_atom, d_leafNodes, d_internalNodes);
}

void BVH::BroadPhase(glm::vec3* d_positions, int arraySize) {
	int blockSize = 256;
	int numBlocks = (arraySize + blockSize - 1) / blockSize;

	CubDebugExit(cudaMemset(d_keysColliding, -1, sizeof(unsigned int) * arraySize * MAX_COLLISIONS));

	queryBVH<<<numBlocks, blockSize>>>(&d_internalNodes[0], d_leafNodes, arraySize, d_positions, d_keysColliding);
}

void BVH::NarrowPhase(glm::vec3* d_oldPositions, glm::vec3* d_positions, glm::vec3* d_impulses, glm::vec3* d_corrections, float* d_collisionsNr, int arraySize) {
	int blockSize = 256;
	int numBlocks = (arraySize + blockSize - 1) / blockSize;

	numBlocks = (arraySize * MAX_COLLISIONS + blockSize - 1) / blockSize;

	narrowPhase<<<numBlocks, blockSize>>>(d_keysColliding, d_oldPositions, d_positions, d_impulses, d_corrections, d_collisionsNr, arraySize * MAX_COLLISIONS);
}

void BVH::NrCollisions(int arraySize) {
	unsigned int* test = new unsigned int[arraySize * MAX_COLLISIONS];
	CubDebugExit(cudaMemcpy(test, d_keysColliding, sizeof(unsigned int) * arraySize * MAX_COLLISIONS, cudaMemcpyDeviceToHost));
	int count = 0;
	for (int i = 0; i < arraySize * MAX_COLLISIONS; i++) {
		if (test[i] != -1) {
			count++;
		}
	}
	printf("Collisions: %d\n", count);
	delete[] test;
}