// This file is distributed under the University of Illinois/NCSA Open Source
// License. See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


#include "QMCWaveFunctions/SPOSet_builder.h"

namespace qmcplusplus
{
template<>
SPOSet* SPOSetBuilder<Devices::KOKKOS>::buildView(const SPOSet* SPOSet_main, int team_size, int member_id)
{
    auto* temp_ptr = dynamic_cast<const EinsplineSPO<Devices::KOKKOS, OHMMS_PRECISION>*>(SPOSet_main);
    auto* spo_view = new EinsplineSPO<Devices::KOKKOS, OHMMS_PRECISION>(*temp_ptr, team_size, member_id);
    return dynamic_cast<SPOSet*>(spo_view);
}


} // namespace qmcplusplus
