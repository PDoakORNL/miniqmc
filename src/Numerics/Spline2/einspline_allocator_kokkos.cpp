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

#include "Numerics/Spline2/einspline_allocator.h"

template<>
einspline_create_multi_UBspline_3d_s_coefs(multi_UBspline_3d_s<Devices::KOKKOS>* restrict spline)
{
  spline->coefs_view = multi_UBspline_3d_s::coefs_view_t("Multi_UBspline_3d_s", Nx, Ny, Nz, N);


  //Check that data layout is as expected
  //
  int strides[4];
  spline->coefs_view.stride(strides);
  if (spline->x_stride != strides[0] || spline->y_stride != strides[1] ||
      spline->z_stride != strides[2] || 1 != strides[3])
    fprintf(stderr,
            "Kokkos View has non-compatible strides %i %i | %i %i | %i %i\n",
            spline->x_stride,
            strides[0],
            spline->y_stride,
            strides[1],
            spline->z_stride,
            strides[2]);

  spline->coefs = spline->coefs_view.data();
}

template<>
einspline_create_multi_UBspline_3d_d_coefs(multi_UBspline_3d_d<Devices::KOKKOS>* restrict spline)
{
  spline->coefs_view = multi_UBspline_3d_d::coefs_view_t("Multi_UBspline_3d_d", Nx, Ny, Nz, N);

  //Check that data layout is as expected
  //
  int strides[4];
  spline->coefs_view.stride(strides);
  if (spline->x_stride != strides[0] || spline->y_stride != strides[1] ||
      spline->z_stride != strides[2] || 1 != strides[3])
    fprintf(stderr,
            "Kokkos View has non-compatible strides %i %i | %i %i | %i %i\n",
            spline->x_stride,
            strides[0],
            spline->y_stride,
            strides[1],
            spline->z_stride,
            strides[2]);

  spline->coefs = spline->coefs_view.data();
}
