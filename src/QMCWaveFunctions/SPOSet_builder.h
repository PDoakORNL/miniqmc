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


#ifndef QMCPLUSPLUS_SINGLEPARTICLEORBITALSET_BUILDER_H
#define QMCPLUSPLUS_SINGLEPARTICLEORBITALSET_BUILDER_H

#include "QMCWaveFunctions/SPOSetImp.h"
#include "QMCWaveFunctions/EinsplineSPO.hpp"

namespace qmcplusplus
{
/// build the einspline SPOSet.
template<Devices DT>
class SPOSetBuilder
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
      EinsplineSPO<DT, OHMMS_PRECISION>* spo_main = new EinsplineSPO<DT, OHMMS_PRECISION>;
      spo_main->set(nx, ny, nz, num_splines, nblocks, tile_size);
      spo_main->setLattice(lattice_b);
      return dynamic_cast<SPOSet*>(spo_main);
  }

  /// build the einspline SPOSet as a view of the main one.
  static SPOSet* buildView(const SPOSet* SPOSet_main, int team_size, int member_id)
  {
      auto* temp_ptr = dynamic_cast<const EinsplineSPO<DT, OHMMS_PRECISION>*>(SPOSet_main);
      auto* spo_view = new EinsplineSPO<DT, OHMMS_PRECISION>(*temp_ptr, team_size, member_id);
      return dynamic_cast<SPOSet*>(spo_view);
  }

  static SPOSet* buildView(const SPOSet& SPOSet_main, int team_size, int member_id)
  {
    return SPOSetBuilder::buildView(useRef, &SPOSet_main, team_size, member_id);
  }
};


} // namespace qmcplusplus
#endif
