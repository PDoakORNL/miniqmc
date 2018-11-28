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

#ifndef QMCPLUSPLUS_EINSPLINE_SPO_DEVICE_H
#define QMCPLUSPLUS_EINSPLINE_SPO_DEVICE_H

#include "Devices.h"
#include "Utilities/Configuration.h"

/** @file
 * CRTP base class for Einspline SPO Devices
 */
namespace qmcplusplus
{
template<class DEVICEIMP, typename T>
class EinsplineSPODevice
{
public:
  using QMCT = QMCTraits;

  void set(int nx, int ny, int nz, int num_splines, int nblocks, bool init_random = true)
  {
    static_cast<DEVICEIMP*>(this)->set(nx,
				       ny,
				       nz,
				       num_splines,
				       nblocks,
				       init_random);
  }
};
} // namespace qmcplusplus

#endif
