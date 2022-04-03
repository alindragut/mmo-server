#include "Memory.cuh"

CachingDeviceAllocator g_allocator(true);  // Caching allocator for device memory
