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
 * @brief Kokkos implementation of EinsplineSPO
 */

#ifndef QMCPLUSPLUS_EINSPLINE_SPO_DEVICE_IMP_KOKKOS_H
#define QMCPLUSPLUS_EINSPLINE_SPO_DEVICE_IMP_KOKKOS_H

#include "clean_inlining.h"
#include <impl/Kokkos_Timer.hpp>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include "Utilities/Configuration.h"
#ifdef KOKKOS_ENABLE_CUDA
#include "cublas_v2.h"
#include "cusolverDn.h"
#endif

namespace qmcpluplus
{
template<>
class EinsplineSPODeviceImp<Devices::KOKKOS>
    : public EinsplineSPODevice<DeterminantDeviceImp<Devices::KOKKOS>>
{
  struct EvaluateVGHTag
  {};
  struct EvaluateVTag
  {};
  typedef Kokkos::TeamPolicy<Kokkos::Serial, EvaluateVGHTag> policy_vgh_serial_t;
  typedef Kokkos::TeamPolicy<EvaluateVGHTag> policy_vgh_parallel_t;
  typedef Kokkos::TeamPolicy<Kokkos::Serial, EvaluateVTag> policy_v_serial_t;
  typedef Kokkos::TeamPolicy<EvaluateVTag> policy_v_parallel_t;

  typedef typename policy_vgh_serial_t::member_type team_vgh_serial_t;
  typedef typename policy_vgh_parallel_t::member_type team_vgh_parallel_t;
  typedef typename policy_v_serial_t::member_type team_v_serial_t;
  typedef typename policy_v_parallel_t::member_type team_v_parallel_t;

  using vContainer_type = Kokkos::View<T*>;
  using gContainer_type = Kokkos::View<T * [3], Kokkos::LayoutLeft>;
  using hContainer_type = Kokkos::View<T * [6], Kokkos::LayoutLeft>;
  using lattice_type    = CrystalLattice<T, 3>;

  Kokkos::View<spline_type*> einsplines;
  Kokkos::View<vContainer_type*> psi;
  Kokkos::View<gContainer_type*> grad;
  Kokkos::View<hContainer_type*> hess;

  //Copy Constructor only supports KOKKOS to KOKKOS
  EinsplineSPODeviceImp(const EinsplineSPODevice<EinsplineSPODeviceImp<Devices::KOKKOS>>& in,
                        int team_size,
                        int member_id)
  {
    nSplinesSerialThreshold_V   = in.nSplinesSerialThreshold_V;
    nSplinesSerialThreshold_VGH = in.nSplinesSerialThreshold_VGH;
    nSplines                    = in.nSplines;
    nSplinesPerBlock            = in.nSplinesPerBlock;
    nBlocks                     = (in.nBlocks + team_size - 1) / team_size;
    firstBlock                  = nBlocks * member_id;
    lastBlock                   = std::min(in.nBlocks, nBlocks * (member_id + 1));
    nBlocks                     = lastBlock - firstBlock;
    einsplines = Kokkos::View<spline_type*>("einsplines", nBlocks);
    for (int i = 0, t = firstBlock; i < nBlocks; ++i, ++t)
      einsplines(i) = in.einsplines(t);
    resize();
  }
};

} // namespace qmcpluplus

#endif
