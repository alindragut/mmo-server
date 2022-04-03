#pragma once

#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>
#define CUB_STDERR

using namespace cub;

extern CachingDeviceAllocator g_allocator;
