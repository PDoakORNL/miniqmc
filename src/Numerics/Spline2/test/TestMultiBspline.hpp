////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
//
// File created by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_TEST_MULTI_BSPLINE_H
#define QMCPLUSPLUS_TEST_MULTI_BSPLINE_H

#include "Utilities/RandomGenerator.h"
#include "Numerics/Spline2/BsplineAllocatorCUDA.hpp"

namespace qmcplusplus
{
using namespace einspline;

/** This sets up of cpu_splines and a cuda_spline for testing
  *  Here I assume that while you use one multibspline per block for CPU
  *  For my modified cudaimplementation you do not use a mbspline per block
  */
template<typename T, typename DT>
struct TestMultiBspline
{
  using CpuST  = bspline_traits<Devices::CPU, T, 3>;
  using CudaST = bspline_traits<Devices::CUDA, DT, 3>;

  Allocator<Devices::CPU> cpu_allocator;
  Allocator<Devices::CUDA> cuda_allocator;
  typename CudaST::SplineType* cuda_spline;
  int num_splines_;
  int num_blocks_;
  int splines_per_block_;
  int grid_num_;
  aligned_vector<typename CpuST::SplineType*> cpu_splines;

  TestMultiBspline(int num_splines = 100, int num_blocks = 1, int grid_num = 100)
      : cuda_spline(nullptr), num_splines_(num_splines), num_blocks_(num_blocks), grid_num_(grid_num)
  {
    cpu_splines.resize(num_blocks_);
    for (int i = 0; i < num_blocks_; ++i)
      cpu_splines[i] = nullptr;
    splines_per_block_ = num_splines_ / num_blocks_;
  };

  void create()
  {
    typename CpuST::BCType bc;
    bc_code bCode = bc_code::PERIODIC;
    bc.lCode      = bCode;
    bc.rCode      = bCode;
    bc.lVal       = 0.0;
    bc.rVal       = 10.0;
    Ugrid grid;
    grid.start = 0.0;
    grid.end   = 10.0;
    grid.num   = grid_num_;

    for (int b = 0; b < num_blocks_; ++b)
    {
      int splines_this_block = std::min(splines_per_block_, num_splines_ - b * splines_per_block_);
      typename CpuST::SplineType*& cpu_spline = cpu_splines[b];
      cpu_allocator.allocateMultiBspline(cpu_spline, grid, grid, grid, bc, bc, bc, num_splines_);
      REQUIRE(cpu_spline != nullptr);
      RandomGenerator<T> myrandom(11 + b);
      Array<T, 3> coef_data(cpu_spline->x_grid.num + 3, cpu_spline->y_grid.num + 3,
                            cpu_spline->z_grid.num + 3);
      for (int i = 0; i < splines_this_block; ++i)
      {
        myrandom.generate_uniform(coef_data.data(), coef_data.size());
        cpu_allocator.setCoefficientsForOneOrbital(i, coef_data, cpu_spline);
      }
    }

    T dummyT;
    DT dummyDT;
    cuda_allocator.createMultiBspline(cpu_splines, cuda_spline, dummyT, dummyDT);

    bool valid_cpu_splines = std::all_of(cpu_splines.begin(), cpu_splines.end(),
                                         [](typename CpuST::SplineType*& cpu_spline) {
                                           bool valid_cpu_spline = (cpu_spline != nullptr);
                                           return valid_cpu_spline;
                                         });

    REQUIRE(valid_cpu_splines);
    REQUIRE(cuda_spline != nullptr);
  }

  void destroy()
  {
    cuda_allocator.destroy(cuda_spline);
    std::for_each(cpu_splines.begin(), cpu_splines.end(),
                  [&](typename CpuST::SplineType*& cpu_spline) { cpu_allocator.destroy(cpu_spline); });
  }
};
} // namespace qmcplusplus

#endif