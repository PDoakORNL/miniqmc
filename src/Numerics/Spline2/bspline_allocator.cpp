////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
/** @file bspline_allocator.cpp
 * @brief Implementation of einspline::Allocator member functions
 *
 * Allocator::Policy is not defined precisely yet but is intended to select
 * the specialized allocator.
 */
#include "Numerics/Spline2/bspline_allocator.hpp"
#include "Numerics/Spline2/einspline_allocator.h"


template<Devices D>
void einspline_create_multi_UBspline_3d_s(multi_UBspline_3d_s<D>* spline,
					  Ugrid x_grid,
                                                          Ugrid y_grid,
                                                          Ugrid z_grid,
                                                          BCtype_s xBC,
                                                          BCtype_s yBC,
                                                          BCtype_s zBC,
                                                          int num_splines);

template<Devices D>
void einspline_create_UBspline_3d_s(
				    UBspline_3d_s<D>* spline,
				    Ugrid x_grid, Ugrid y_grid, Ugrid z_grid, BCtype_s xBC, BCtype_s yBC, BCtype_s zBC);

template<Devices D>
void einspline_create_multi_UBspline_3d_d(multi_UBspline_3d_d<D>* spline,
					  Ugrid x_grid,
                                                          Ugrid y_grid,
                                                          Ugrid z_grid,
                                                          BCtype_d xBC,
                                                          BCtype_d yBC,
                                                          BCtype_d zBC,
                                                          int num_splines);

template<Devices D>
void einspline_create_UBspline_3d_d(UBspline_3d_d<D>* spline,
    Ugrid x_grid, Ugrid y_grid, Ugrid z_grid, BCtype_d xBC, BCtype_d yBC, BCtype_d zBC);

namespace qmcplusplus
{
namespace einspline
{
template<Devices D>
Allocator<D>::Allocator() : Policy(0) {}

template<Devices D>
Allocator<D>::~Allocator() {}

template<Devices D>
void Allocator<D>::allocateMultiBspline(multi_UBspline_3d_s<D>* spline,
    Ugrid x_grid, Ugrid y_grid, Ugrid z_grid, BCtype_s xBC, BCtype_s yBC, BCtype_s zBC, int num_splines)
{
  einspline_create_multi_UBspline_3d_s(spline, x_grid, y_grid, z_grid, xBC, yBC, zBC, num_splines);
}

template<Devices D>
void Allocator<D>::allocateMultiBspline(multi_UBspline_3d_d<D>* spline,
    Ugrid x_grid, Ugrid y_grid, Ugrid z_grid, BCtype_d xBC, BCtype_d yBC, BCtype_d zBC, int num_splines)
{
  einspline_create_multi_UBspline_3d_d(spline, x_grid, y_grid, z_grid, xBC, yBC, zBC, num_splines);
}

template<Devices D>
void Allocator<D>::allocateUBspline(UBspline_3d_d<D>* spline,
    Ugrid x_grid, Ugrid y_grid, Ugrid z_grid, BCtype_d xBC, BCtype_d yBC, BCtype_d zBC)
{
  einspline_create_UBspline_3d_d(spline, x_grid, y_grid, z_grid, xBC, yBC, zBC);
}

template<Devices D>
void  Allocator<D>::allocateUBspline(UBspline_3d_s<D>* spline,
    Ugrid x_grid, Ugrid y_grid, Ugrid z_grid, BCtype_s xBC, BCtype_s yBC, BCtype_s zBC)
{
  return einspline_create_UBspline_3d_s(spline, x_grid, y_grid, z_grid, xBC, yBC, zBC);
}


} // namespace einspline
} // namespace qmcplusplus
