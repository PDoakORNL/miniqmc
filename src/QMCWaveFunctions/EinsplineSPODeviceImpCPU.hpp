////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2018 QMCPACK developers.
//
// File developed by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
//
// File created by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-

/**
 * @file
 * @brief CPU implementation of EinsplineSPO
 */

#ifndef QMCPLUSPLUS_EINSPLINE_SPO_DEVICE_IMP_CPU_H
#define QMCPLUSPLUS_EINSPLINE_SPO_DEVICE_IMP_CPU_H

#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>
#include <array>
#include <memory>
#include "Devices.h"
#include "clean_inlining.h"
#include "Numerics/Containers.h"
#include "Utilities/SIMD/allocator.hpp"
#include "Utilities/Configuration.h"
#include "Utilities/NewTimer.h"
#include "Utilities/RandomGenerator.h"
#include "QMCWaveFunctions/EinsplineSPODevice.hpp"
#include "QMCWaveFunctions/EinsplineSPODeviceImp.hpp"
#include "QMCWaveFunctions/EinsplineSPOParams.h"
#include "Numerics/Spline2/SplineBundle.hpp"
#include "Numerics/Spline2/bspline_traits.hpp"
#include "Numerics/Spline2/bspline_allocator.hpp"
#include "Numerics/Spline2/MultiBsplineFuncs.hpp"

namespace qmcplusplus
{
template<typename T>
class EinsplineSPODeviceImp<Devices::CPU, T>
    : public EinsplineSPODevice<EinsplineSPODeviceImp<Devices::CPU, T>, T>
{
  using QMCT = QMCTraits;
  /// define the einspline data object type
  using spline_type     = typename bspline_traits<Devices::CPU, T, 3>::SplineType;
  using vContainer_type = aligned_vector<T>;
  using gContainer_type = VectorSoAContainer<T, 3>;
  using hContainer_type = VectorSoAContainer<T, 6>;
  using lattice_type    = CrystalLattice<T, 3>;

  /// use allocator
  einspline::Allocator<Devices::CPU> myAllocator;
  /// compute engine
  MultiBsplineFuncs<Devices::CPU, T> compute_engine;

  //using einspline_type = spline_type*;
  std::shared_ptr<SplineBundle<Devices::CPU, T>> our_einsplines;
  
  //  aligned_vector<vContainer_type> psi;
  aligned_vector<vContainer_type> psi;
  aligned_vector<gContainer_type> grad;
  aligned_vector<hContainer_type> hess;
  EinsplineSPOParams<T> esp;
public:
  EinsplineSPODeviceImp()
  {
    //std::cout << "EinsplineSPODeviceImpCPU() called" << '\n';
    esp.nBlocks    = 0;
    esp.nSplines   = 0;
    esp.firstBlock = 0;
    esp.lastBlock  = 0;
    our_einsplines = nullptr;
    psi            = {};
    grad           = {};
    hess           = {};
  }

  /** CPU to CPU Constructor
   */
  EinsplineSPODeviceImp(const EinsplineSPODeviceImp<Devices::CPU, T>& in)
  {
    //std::cout << "EinsplineSPODeviceImpCPU Fat Copy constructor called" << '\n';
    const EinsplineSPOParams<T>& inesp = in.getParams();
    esp.nSplinesSerialThreshold_V      = inesp.nSplinesSerialThreshold_V;
    esp.nSplinesSerialThreshold_VGH    = inesp.nSplinesSerialThreshold_VGH;
    esp.nSplines                       = inesp.nSplines;
    esp.nSplinesPerBlock               = inesp.nSplinesPerBlock;
    esp.nBlocks                        = inesp.nBlocks;
    esp.firstBlock                     = 0;
    esp.lastBlock                      = inesp.nBlocks;
    esp.lattice                        = inesp.lattice;
    our_einsplines = in.our_einsplines;
    resize();
  }

  /** "Fat" Copy Constructor only supports CPU to CPU
   */
  EinsplineSPODeviceImp(const EinsplineSPODeviceImp<Devices::CPU, T>& in,
                        int team_size,
                        int member_id)
  {
    std::cout << "EinsplineSPODeviceImpCPU Fat Copy constructor called" << '\n';
    const EinsplineSPOParams<T>& inesp = in.getParams();
    esp.nSplinesSerialThreshold_V      = inesp.nSplinesSerialThreshold_V;
    esp.nSplinesSerialThreshold_VGH    = inesp.nSplinesSerialThreshold_VGH;
    esp.nSplines                       = inesp.nSplines;
    esp.nSplinesPerBlock               = inesp.nSplinesPerBlock;
    esp.nBlocks                        = (inesp.nBlocks + team_size - 1) / team_size;
    esp.firstBlock                     = esp.nBlocks * member_id;
    esp.lastBlock                      = std::min(inesp.nBlocks, esp.nBlocks * (member_id + 1));
    esp.nBlocks                        = esp.lastBlock - esp.firstBlock;
    esp.lattice                        = inesp.lattice;
    our_einsplines = in.our_einsplines;
    resize();
  }

  /// destructors
  ~EinsplineSPODeviceImp()
  {
  }

  /// resize the containers
  void resize()
  {
    if (esp.nBlocks > 0)
    {
      this->psi.resize(this->esp.nBlocks);
      this->grad.resize(esp.nBlocks);
      this->hess.resize(esp.nBlocks);
      for (int i = 0; i < esp.nBlocks; ++i)
      {
        this->psi[i].resize(esp.nSplinesPerBlock);
        this->grad[i].resize(esp.nSplinesPerBlock);
        this->hess[i].resize(esp.nSplinesPerBlock);
      }
    }
  }

  void set_i(int nx, int ny, int nz, int num_splines, int nblocks, bool init_random = true)
  {
    this->esp.nSplines         = num_splines;
    this->esp.nBlocks          = nblocks;
    this->esp.nSplinesPerBlock = num_splines / nblocks;
    this->esp.firstBlock       = 0;
    this->esp.lastBlock        = esp.nBlocks;
    if (our_einsplines == nullptr)
    {
      TinyVector<int, 3> ng(nx, ny, nz);
      QMCT::PosType start(0);
      QMCT::PosType end(1);
      our_einsplines = std::make_shared<SplineBundle<Devices::CPU,T>>();
      aligned_vector<spline_type*>& einsplines = our_einsplines->einsplines;
      einsplines.resize(esp.nBlocks);
      RandomGenerator<T> myrandom(11);
      Array<T, 3> coef_data(nx + 3, ny + 3, nz + 3);
      for (int i = 0; i < esp.nBlocks; ++i)
      {
        this->myAllocator
            .createMultiBspline(einsplines[i], T(0), start, end, ng, PERIODIC, esp.nSplinesPerBlock);
        if (init_random)
        {
          for (int j = 0; j < esp.nSplinesPerBlock; ++j)
          {
            // Generate different coefficients for each orbital
            myrandom.generate_uniform(coef_data.data(), coef_data.size());
            myAllocator.setCoefficientsForOneOrbital(j, coef_data, einsplines[i]);
          }
        }
      }
    }
    resize();
  }

  const EinsplineSPOParams<T>& getParams_i() const { return this->esp; }

  void* getEinspline_i(int i) const { return our_einsplines->einsplines[i]; }

  void setLattice_i(const Tensor<T, 3>& lattice) { esp.lattice.set(lattice); }

  inline void evaluate_v_i(const QMCT::PosType& p)
  {
    auto u = esp.lattice.toUnit_floor(p);
    std::vector<std::array<T,3>> pos = {{u[0],u[1],u[2]}}; 
    for (int i = 0; i < esp.nBlocks; ++i)
      compute_engine.evaluate_v(our_einsplines->einsplines[i], pos, psi[i].data(), esp.nSplinesPerBlock);
  }

  inline void evaluate_vgh_i(const QMCT::PosType& p)
  {
    auto u = esp.lattice.toUnit_floor(p);
    std::vector<std::array<T,3>> pos = {{u[0],u[1],u[2]}}; 
    for (int i = 0; i < esp.nBlocks; ++i)
    {
      compute_engine.evaluate_vgh(our_einsplines->einsplines[i],
                                  pos,
                                  psi[i].data(),
                                  grad[i].data(),
                                  hess[i].data(),
                                  esp.nSplinesPerBlock);
    }
  }

  void evaluate_vgl_i(const QMCT::PosType& p)
  {
    auto u = esp.lattice.toUnit_floor(p);
    std::vector<std::array<T,3>> pos = {{u[0],u[1],u[2]}}; 
    for (int i = 0; i < esp.nBlocks; ++i)
      compute_engine.evaluate_vgl(our_einsplines->einsplines[i],
                                  pos,
                                  psi[i].data(),
                                  grad[i].data(),
                                  hess[i].data(),
                                  esp.nSplinesPerBlock);
  }

  T getPsi_i(int ib, int n) { return psi[ib][n]; }

  T getGrad_i(int ib, int n, int m) { return grad[ib].data(m)[n]; }

  T getHess_i(int ib, int n, int m) { return hess[ib].data(m)[n]; }
};

extern template class EinsplineSPODeviceImp<Devices::CPU, float>;
extern template class EinsplineSPODeviceImp<Devices::CPU, double>;

} // namespace qmcplusplus

#endif
