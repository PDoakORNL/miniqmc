////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp.
// Amrita Mathuriya, amrita.mathuriya@intel.com,
//    Intel Corp.
//
// File created by:
// Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp.
// ////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
/** @file
 */
#ifndef QMCPLUSPLUS_EINSPLINE_SPO_HPP
#define QMCPLUSPLUS_EINSPLINE_SPO_HPP
#include <Utilities/Configuration.h>
#include <Utilities/NewTimer.h>
#include <Particle/ParticleSet.h>
#include "Devices.h"
#include <Numerics/Spline2/bspline_allocator.hpp>
#include <Numerics/Spline2/MultiBspline.hpp>
#include <Utilities/SIMD/allocator.hpp>
#include "Numerics/OhmmsPETE/OhmmsArray.h"
#include "QMCWaveFunctions/EinsplineSPOParams.h"
#include "QMCWaveFunctions/EinsplineSPODevice.hpp"
#include "QMCWaveFunctions/EinsplineSPODeviceImp.hpp"
#include "QMCWaveFunctions/SPOSetImp.h"
#include <iostream>

namespace qmcplusplus
{
template <Devices DT, typename T>
class EinsplineSPO : public SPOSetImp<DT>
{
public:

  // Global Type Aliases
  using QMCT = QMCTraits;
  using PosType = QMCT::PosType;
  using lattice_type    = CrystalLattice<T, 3>;
  // Base SPOSetImp
  using BaseSPO = SPOSetImp<DT>;
  
  /// Timer
  NewTimer* timer;

  /// default constructor
  EinsplineSPO()
  {
    timer = TimerManager.createTimer("Single-Particle Orbitals", timer_level_fine);
  }
  /// disable copy constructor
  EinsplineSPO(const EinsplineSPO& in) = default;
  /// disable copy operator
  EinsplineSPO& operator=(const EinsplineSPO& in) = delete;

  /** copy constructor
   * @param in EinsplineSPO
   * @param team_size number of members in a team
   * @param member_id id of this member in a team
   *
   * Create a view of the big object. A simple blocking & padding  method.
   */
  EinsplineSPO(const EinsplineSPO& in, int team_size, int member_id)
    : einspline_spo_device(in.einspline_spo_device, team_size, member_id)
  {
    
    // einsplines.resize(nBlocks);
    timer = TimerManager.createTimer("Single-Particle Orbitals", timer_level_fine);
  }

  /// destructors
  ~EinsplineSPO()
  {
    //Note the change in garbage collection here.  The reason for doing this is that by
    //changing einsplines to a view, it's more natural to work by reference than by raw pointer.
    // To maintain current interface, redoing the input types of allocate and destroy to call by references
    //  would need to be propagated all the way down.
    // However, since we've converted the large chunks of memory to views, garbage collection is
    // handled automatically.  Thus, setting the spline_type objects to empty views lets Kokkos handle the Garbage collection.

    
    //    for (int i = 0; i < nBlocks; ++i)
    //      myAllocator.destroy(einsplines(i));
  }


  // fix for general num_splines
  void set(int nx, int ny, int nz, int num_splines, int nblocks, bool init_random = true)
  {
    einspline_spo_device.set(nx, ny, nz, num_splines, nblocks, init_random);
  }

  /** evaluate psi */
  inline void evaluate_v(const PosType& p)
  {
    einspline_spo_device.evaluate_v(p);
  }

  /** evaluate psi probably for synced walker */
  inline void evaluate_v_pfor(const PosType& p)
  {
    einspline_spo_device.evaluate_v_pfor(p);
  }

  /** evaluate psi, grad and lap */
  inline void evaluate_vgl(const PosType& p)
  {
    einspline_spo_device.evaluate_vgl(p);
  }

  /** evaluate psi, grad and hess */
  inline void evaluate_vgh(const PosType& p)
  {
    einspline_spo_device.evaluate_vgh(p);
  }

  void print(std::ostream& os)
  {
    os << einspline_spo_device;
  }

  void setLattice(const Tensor<T, 3>& lattice)
  {
    einspline_spo_device.setLattice(lattice);
  }

  const EinsplineSPOParams<T>& getParams()
  {
    return einspline_spo_device.getParams();
  }
private:
  EinsplineSPODevice<EinsplineSPODeviceImp<DT, T>, T> einspline_spo_device;
};

// template<Devices DT, typename T>
// KOKKOS_INLINE_FUNCTION void EinsplineSPO<DT, T>::
//     operator()(const EvaluateVTag&, const team_v_serial_t& team) const
// {
// }

// template<Devices DT, typename T>
// KOKKOS_INLINE_FUNCTION void EinsplineSPO<DT, T>::
//     operator()(const EvaluateVTag&, const team_v_parallel_t& team) const
// {
// }
  
// template<typename T>
// KOKKOS_INLINE_FUNCTION void EinsplineSPO<Devices::KOKKOS, T>::
// operator()(const typename EvaluateVTag&,
// 	   const typename EinsplineSPO<Devices::KOKKOS, T>::team_v_serial_t& team) const
// {
//   int block               = team.league_rank();
//   auto u                  = Lattice.toUnit_floor(tmp_pos);
//   einsplines(block).coefs = einsplines(block).coefs_view.data();
//   compute_engine
//       .evaluate_v(team, &einsplines(block), u[0], u[1], u[2], psi(block).data(), psi(block).extent(0));
// }

// template<typename T>
// KOKKOS_INLINE_FUNCTION void EinsplineSPO<Devices::KOKKOS, T>::
// operator()(const typename EinsplineSPO<Devices::KOKKOS, T>::eEvaluateVTag&,
// 	   const typename EinsplineSPO<Devices::KOKKOS, T>::team_v_parallel_t& team) const
// {
//   int block               = team.league_rank();
//   auto u                  = Lattice.toUnit_floor(tmp_pos);
//   einsplines(block).coefs = einsplines(block).coefs_view.data();
//   compute_engine
//       .evaluate_v(team, &einsplines(block), u[0], u[1], u[2], psi(block).data(), psi(block).extent(0));
// }

} // namespace qmcplusplus

#endif