#ifndef MULTI_BSPLINE_STRUCTS_KOKKOS_H
#define MULTI_BSPLINE_STRUCTS_KOKKOS_H

#define MULTI_UBSPLINE_KOKKOS_VIEW_DEF \
  typedef Kokkos::View<float****, Kokkos::LayoutRight> coefs_view_t;\
  coefs_view_t coefs_view;

template<>
struct multi_UBspline_3d_s<Devices::KOKKOS> : public multi_UBspline_3d_s_common
{
  MULTI_UBSPLINE_KOKKOS_VIEW_DEF
};

#ifdef QMC_USE_KOKKOS
#endif

template<>
struct multi_UBspline_3d_d<Devices::KOKKOS> : public multi_UBspline_3d_d_common
{
  MULTI_UBSPLINE_KOKKOS_VIEW_DEF
};



  
#endif
