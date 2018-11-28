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

#ifndef QMCPLUSPLUS_EINSPLINE_SPO_DEVICE_IMP_H
#define QMCPLUSPLUS_EINSPLINE_SPO_DEVICE_IMP_H

#include "Devices.h" 

/** @file
 * Here EinsplineSPO device implementations are included
 */

namespace qmcplusplus
{
template<Devices DT, typename T>
class EinsplineSPODeviceImp
{
};
}

#include "QMCWaveFunctions/EinsplineSPODeviceImpCPU.hpp"
#ifdef QMC_USE_KOKKOS
#include "QMCWaveFunctions/EinsplineSPODeviceImpKOKKOS.hpp"
#endif

namespace qmcplusplus
{
extern template class EinsplineSPODeviceImp<Devices::CPU, float>;
extern template class EinsplineSPODeviceImp<Devices::CPU, double>;
#ifdef QMC_USE_KOKKOS
extern template class EinsplineSPODeviceImp<Devices::KOKKOS, float>;
extern template class EinsplineSPODeviceImp<Devices::KOKKOS, double>;
#endif
}


#endif
