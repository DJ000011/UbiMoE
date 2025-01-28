#ifndef __GELU_HPP__
#define __GELU_HPP__
#include <gmp.h>
#define __gmp_const const
#include "dcl.hpp"

fm_t gelu(fm_t x);

template<size_t N>
hls::vector<fm_t, N> gelu(hls::vector<fm_t, N> x);
#include "gelu.cpp"
#endif
