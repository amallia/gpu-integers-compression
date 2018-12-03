#pragma once

namespace utils {

static void delta_encode(uint32_t *in, size_t n) {
  for (size_t i = 1; i < n; ++i) {
    in[i] -= in[i-1];
  }
}

}