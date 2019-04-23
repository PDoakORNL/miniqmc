// This file is distributed under the University of Illinois/NCSA Open Source
// License. See LICENSE file in top directory for details.
//
// Copyright (c) 2018 QMCPACK developers.
//
// File developed by: Peter Doak, doakpw@ornl.gov, Oak Ridege National Laboratory
//                    Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////

/** @file
 *  This specializes the SPOSetBuilder for the reference implementation,
 *  all the others share a common architecture but this attempts to presever the "reference" code
 *  as close as possible to its original state
 */

#ifndef QMCPLUSPLUS_SINGLEPARTICLEORBITALSET_BUILDER_REF_H
#define QMCPLUSPLUS_SINGLEPARTICLEORBITALSET_BUILDER_REF_H

#include "QMCWaveFunctions/SPOSetImp.h"
#include "QMCWaveFunctions/EinsplineSPO.hpp"
#include "QMCWaveFunctions/einspline_spo_ref.hpp"

namespace qmcplusplus
{

template<>
class SPOSetBuilder<Devices::REFERENCE>
{
public:
  static SPOSet* build(                       int nx,
                       int ny,
                       int nz,
                       int num_splines,
                       int nblocks,
                       int tile_size,
                       const Tensor<OHMMS_PRECISION, 3>& lattice_b,
                       bool init_random = true)
  {
      miniqmcreference::EinsplineSPO_ref<OHMMS_PRECISION>* spo_main =
      new miniqmcreference::EinsplineSPO_ref<OHMMS_PRECISION>;
      spo_main->set(nx, ny, nz, num_splines, nblocks);
      spo_main->Lattice.set(lattice_b);
      return dynamic_cast<SPOSet*>(spo_main);
  }

  /// build the einspline SPOSet as a view of the main one.
  static SPOSet* buildView(const SPOSet* SPOSet_main, int team_size, int member_id)
  {
      auto* temp_ptr = dynamic_cast<const miniqmcreference::EinsplineSPO_ref<OHMMS_PRECISION>*>(SPOSet_main);
      auto* spo_view = new miniqmcreference::EinsplineSPO_ref<OHMMS_PRECISION>(*temp_ptr, team_size, member_id);
      return dynamic_cast<SPOSet*>(spo_view);
  }

  static SPOSet* buildView(const SPOSet& SPOSet_main, int team_size, int member_id)
  {
    return SPOSetBuilder::buildView(&SPOSet_main, team_size, member_id);
  }
};
