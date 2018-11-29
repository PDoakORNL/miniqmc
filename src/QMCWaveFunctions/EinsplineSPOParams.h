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

#ifndef QMCPLUSPLUS_EINSPLINE_SPO_PARAMS_H
#define QMCPLUSPLUS_EINSPLINE_SPO_PARAMS_H

namespace qmcplusplus
{
template<typename T>
struct EinsplineSPOParams
{
  /// number of blocks
  int nBlocks = 0;
  /// first logical block index
  int firstBlock = 0;
  /// last logical block index
  int lastBlock = 0;
  /// number of splines
  int nSplines = 0;
  /// number of splines per block
  int nSplinesPerBlock = 0;
  int nSplinesSerialThreshold_V = 0;
  int nSplinesSerialThreshold_VGH = 0;
  

  /// if true, responsible for cleaning up einsplines
  bool Owner = false;
  /// if true, is copy.  For reference counting & clean up in Kokkos.
  bool is_copy = false;

  CrystalLattice<T, 3> lattice;
};
  
}

#endif
