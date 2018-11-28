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
 * @brief Shared definitions for EinsplineSPODevices
 */

#ifndef QMCPLUSPLUS_EINSPLINE_SPO_DEVICE_COMMON_H
#define QMCPLUSPLUS_EINSPLINE_SPO_DEVICE_COMMON_H

namespace qmcplusplus
{
template<typename T>
class EinsplineSPODeviceCommon
{
  /// number of blocks
  int nBlocks = 0;
  /// first logical block index
  int firstBlock = 0;
  /// last gical block index
  int lastBlock = 0;
  /// number of splines
  int nSplines = 0;
  /// number of splines per block
  int nSplinesPerBlock;
  int nSplinesSerialThreshold_V;
  int nSplinesSerialThreshold_VGH;
  

  /// if true, responsible for cleaning up einsplines
  bool Owner = false;
  /// if true, is copy.  For reference counting & clean up in Kokkos.
  bool is_copy = false;

  CrystalLattice<T, 3> lattice;
  
};
  
}

#endif //QMCPLUSPLUS_EINSPLINE_SPO_DEVICE_IMP_CPU_H
