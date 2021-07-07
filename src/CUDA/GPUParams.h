//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Ying Wai Li, yingwaili@ornl.gov, Oak Ridge National Laboratory
//                    Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#ifndef GPU_MISC_H
#define GPU_MISC_H

#include <cstdlib>
#include <cstdio>

#include <cuda_runtime_api.h>
#include <vector>
#include <thread>

#include <cublas_v2.h>

#define COALLESCED_SIZE 16

//#define

struct Gpu
{
public:
  static Gpu& get()
  {
    static thread_local Gpu instance;
    return instance;
  }

private:
  Gpu()
  {
    int device;
    cudaDeviceProp prop;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    initCUDAStreams();
  }

  void initCUDAStreams()
  {
    kernelStreams.resize(2);
    memoryStreams.resize(2);
    for (int i = 0; i < 2; ++i)
    {
      cudaError_t err = cudaStreamCreate(&(kernelStreams[i]));
      err             = cudaStreamCreate(&(memoryStreams[i]));
    }
    //cudaStreamCreate(&kernelStream);
    //cudaStreamCreate(&memoryStream);
  }

  Gpu(const Gpu&) = delete;
  Gpu& operator=(const Gpu&) = delete;

public:
  std::vector<cudaStream_t> kernelStreams;
  std::vector<cudaStream_t> memoryStreams;

  cudaEvent_t syncEvent;

  cudaEvent_t gradientSyncDiracEvent;
  cudaEvent_t gradientSyncOneBodyEvent;
  cudaEvent_t gradientSyncTwoBodyEvent;

  cudaEvent_t ratioSyncDiracEvent;
  cudaEvent_t ratioSyncOneBodyEvent;
  cudaEvent_t ratioSyncTwoBodyEvent;

  cublasHandle_t cublasHandle;
  int device;
  cudaDeviceProp prop;

  size_t MaxGPUSpineSizeMB;
  int rank;
  int relative_rank;                     // relative rank number on the node the rank is on, counting starts at zero
  int device_group_size;                 // size of the lists below
  bool cudamps;                          // is set to true if Cuda MPS service is running
  std::vector<int> device_group_numbers; // on node list of GPU device numbers with respect to relative rank number
  std::vector<int>
      device_rank_numbers; // on node list of MPI rank numbers (absolute) with respect to relative rank number


  void initCUDAEvents();
  void initCublas();

  void finalizeCUDAStreams();
  void finalizeCUDAEvents();
  void finalizeCublas();

  void synchronize();

  void streamsSynchronize();
};

#endif