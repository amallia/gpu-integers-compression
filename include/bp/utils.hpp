#pragma once

namespace utils {

static void delta_encode(uint32_t *in, size_t n) {
    for (size_t i = n - 1; i > 0; --i) {
      in[i] -= in[i - 1];
    }
}

}