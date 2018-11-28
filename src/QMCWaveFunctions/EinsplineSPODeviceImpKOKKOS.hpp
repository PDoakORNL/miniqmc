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

namespace qmcplusplus
{
template<typename T>
class EinsplineSPODeviceImp<Devices::KOKKOS, T>
  : public EinsplineSPODeviceCommon<T>,
    public EinsplineSPODevice<EinsplineSPODeviceImp<Devices::KOKKOS,T>, T>
{
  struct EvaluateVGHTag
  {};
  struct EvaluateVTag
  {};
  using QMCT = QMCTraits;
  using ESDC = EinsplineSPODeviceCommon<T>;
  using spline_type     = typename bspline_traits<Devices::KOKKOS, T, 3>::SplineType;

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
  EinsplineSPODeviceImp(const EinsplineSPODevice<EinsplineSPODeviceImp<Devices::KOKKOS, T>, T>& in,
                        int team_size,
                        int member_id)
  {
    ESDC::nSplinesSerialThreshold_V   = in.nSplinesSerialThreshold_V;
    ESDC::nSplinesSerialThreshold_VGH = in.nSplinesSerialThreshold_VGH;
    ESDC::nSplines                    = in.nSplines;
    ESDC::nSplinesPerBlock            = in.nSplinesPerBlock;
    ESDC::nBlocks                     = (in.nBlocks + team_size - 1) / team_size;
    ESDC::firstBlock                  = ESDC::nBlocks * member_id;
    ESDC::lastBlock                   = std::min(in.nBlocks, ESDC::nBlocks * (member_id + 1));
    ESDC::nBlocks                     = ESDC::lastBlock - ESDC::firstBlock;
    einsplines                  = Kokkos::View<spline_type*>("einsplines", ESDC::nBlocks);
    for (int i = 0, t = ESDC::firstBlock; i < ESDC::nBlocks; ++i, ++t)
      einsplines(i) = in.einsplines(t);
    resize();
  }

  /// resize the containers
  void resize()
  {
    //    psi.resize(nBlocks);
    //    grad.resize(nBlocks);
    //    hess.resize(nBlocks);

    psi  = Kokkos::View<vContainer_type*>("Psi", ESDC::nBlocks);
    grad = Kokkos::View<gContainer_type*>("Grad", ESDC::nBlocks);
    hess = Kokkos::View<hContainer_type*>("Hess", ESDC::nBlocks);

    for (int i = 0; i < ESDC::nBlocks; ++i)
    {
      //psi[i].resize(nSplinesPerBlock);
      //grad[i].resize(nSplinesPerBlock);
      //hess[i].resize(nSplinesPerBlock);

      //Using the "view-of-views" placement-new construct.
      new (&psi(i)) vContainer_type("psi_i", ESDC::nSplinesPerBlock);
      new (&grad(i)) gContainer_type("grad_i", ESDC::nSplinesPerBlock);
      new (&hess(i)) hContainer_type("hess_i", ESDC::nSplinesPerBlock);
    }
  }

  ~EinsplineSPODeviceImp()
  {
    if (!ESDC::is_copy)
    {
      einsplines = Kokkos::View<spline_type*>();
      for (int i = 0; i < psi.extent(0); i++)
      {
        psi(i)  = vContainer_type();
        grad(i) = gContainer_type();
        hess(i) = hContainer_type();
      }
      psi  = Kokkos::View<vContainer_type*>();
      grad = Kokkos::View<gContainer_type*>();
      hess = Kokkos::View<hContainer_type*>();
    }
  }

  void set(int nx, int ny, int nz, int num_splines, int nblocks, bool init_random = true)
  {
    ESDC::nSplines         = num_splines;
    ESDC::nBlocks          = nblocks;
    ESDC::nSplinesPerBlock = num_splines / nblocks;
    ESDC::firstBlock       = 0;
    ESDC::lastBlock        = nblocks;
    if (einsplines.extent(0) == 0)
    {
      ESDC::Owner = true;
      TinyVector<int, 3> ng(nx, ny, nz);
      QMCT::PosType start(0);
      QMCT::PosType end(1);

      //    einsplines.resize(nBlocks);
      einsplines = Kokkos::View<spline_type*>("einsplines", nBlocks);

      RandomGenerator<T> myrandom(11);
      //Array<T, 3> coef_data(nx+3, ny+3, nz+3);
      Kokkos::View<T***> coef_data("coef_data", nx + 3, ny + 3, nz + 3);

      for (int i = 0; i < ESDC::nBlocks; ++i)
      {
        myAllocator.createMultiBspline(&einsplines(i), T(0), start, end, ng, PERIODIC, nSplinesPerBlock);
        if (init_random)
        {
          for (int j = 0; j < nSplinesPerBlock; ++j)
          {
            // Generate different coefficients for each orbital
            myrandom.generate_uniform(coef_data.data(), coef_data.extent(0));
            myAllocator.setCoefficientsForOneOrbital(j, coef_data, &einsplines(i));
          }
        }
      }
    }
    resize();
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const EvaluateVTag&, const team_v_serial_t& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(const EvaluateVTag&, const team_v_parallel_t& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(const EvaluateVGHTag&, const team_vgh_parallel_t& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(const EvaluateVGHTag&, const team_vgh_serial_t& team) const;
};

} // namespace qmcpluplus

#endif
