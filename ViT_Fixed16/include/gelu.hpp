#ifndef __GELU_HPP__
#define __GELU_HPP__

#include "dcl.hpp"

#include <hls_math.h>
#include <hls_vector.h>

fm_t gelu(fm32_t x);

template<size_t N>
hls::vector<fm_t, N> gelu(hls::vector<fm32_t, N> x);

#include "../src/gelu.cpp"
#endif
