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

#include "Devices.h"
#include "clean_inlining.h"
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include "Utilities/Configuration.h"
#include "Utilities/NewTimer.h"
#include "QMCWaveFunctions/EinsplineSPODeviceImp.hpp"
#include "QMCWaveFunctions/EinsplineSPODevice.hpp"
#include "QMCWaveFunctions/EinsplineSPOParams.h"
#include "Numerics/Spline2/bspline_traits.hpp"

namespace qmcplusplus
{

template<typename T>
class EinsplineSPODeviceImp<Devices::CPU, T>
  : public EinsplineSPODevice<EinsplineSPODeviceImp<Devices::CPU, T>, T>
{

  /// define the einsplie data object type
  using spline_type     = typename bspline_traits<Devices::CPU, T, 3>::SplineType;
  using vContainer_type = aligned_vector<T>;
  using gContainer_type = VectorSoAContainer<T, 3>;
  using hContainer_type = VectorSoAContainer<T, 6>;
  using lattice_type    = CrystalLattice<T, 3>;

  /// use allocator
  einspline::Allocator<Devices::CPU> myAllocator;
  /// compute engine
  MultiBspline<Devices::CPU, T> compute_engine;

  aligned_vector<spline_type*> einsplines;
  aligned_vector<vContainer_type> psi;
  aligned_vector<gContainer_type> grad;
  aligned_vector<hContainer_type> hess;

  EinsplineSPOParams<T> esp;
  
  //Copy Constructor only supports CPU to CPU
  EinsplineSPODeviceImp(const EinsplineSPODevice<EinsplineSPODeviceImp<Devices::CPU, T>, T>& in,
                        int team_size,
                        int member_id)
  {
    esp.nSplinesSerialThreshold_V   = in.nSplinesSerialThreshold_V;
    esp.nSplinesSerialThreshold_VGH = in.nSplinesSerialThreshold_VGH;
    esp.nSplines                    = in.nSplines;
    esp.nSplinesPerBlock            = in.nSplinesPerBlock;
    esp.nBlocks                     = (in.nBlocks + team_size - 1) / team_size;
    esp.firstBlock                  = esp.nBlocks * member_id;
    esp.lastBlock                   = std::min(in.nBlocks, esp.nBlocks * (member_id + 1));
    esp.nBlocks                     = esp.lastBlock - esp.firstBlock;
    einsplines.resize(esp.nBlocks);
    for (int i = 0, t = esp.firstBlock; i < esp.nBlocks; ++i, ++t)
      einsplines[i] = in.einsplines[t];
    resize();
  }

    /// destructors
  ~EinsplineSPODeviceImp()
  {
    if (esp.Owner)
      for (int i = 0; i < esp.nBlocks; ++i)
        myAllocator.destroy(einsplines[i]);
  }

  /// resize the containers
  void resize()
  {
    psi.resize(esp.nBlocks);
    grad.resize(esp.nBlocks);
    hess.resize(esp.nBlocks);
    for (int i = 0; i < esp.nBlocks; ++i)
    {
      psi[i].resize(esp.nSplinesPerBlock);
      grad[i].resize(esp.nSplinesPerBlock);
      hess[i].resize(esp.nSplinesPerBlock);
    }
  }

};

} // namespace qmcpluplus

#endif
