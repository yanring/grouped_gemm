/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

#include <string>
#include <iostream>

static const char* _cudaGetErrorEnum(cudaError_t error)
{
    return cudaGetErrorString(error);
}

template<typename T>
void check(T result, char const* const func, const char* const file, int const line)
{
    if (result) {
        throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") + (_cudaGetErrorEnum(result)) + " "
                                 + file + ":" + std::to_string(line) + " \n");
    }
}
#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)

inline int getSMVersion()
{
    int device{-1};
    check_cuda_error(cudaGetDevice(&device));
    int sm_major = 0;
    int sm_minor = 0;
    check_cuda_error(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
    check_cuda_error(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
    return sm_major * 10 + sm_minor;
}
