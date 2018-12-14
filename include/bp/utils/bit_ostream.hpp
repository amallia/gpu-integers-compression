/**
 * Copyright 2018-present Antonio Mallia <me@antoniomallia.it>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "utils.hpp"

class bit_ostream {
   public:
    bit_ostream(uint8_t *buf) : m_buf(reinterpret_cast<uint32_t *>(buf)), m_size(0) {}

    void write(uint32_t bits, uint32_t len) {
        if (!len)
            return;
        uint32_t pos_in_word = m_size % 32;

        m_size += len;
        if (pos_in_word == 0) {
            *m_buf = bits;
            if (len == 32) {
                m_buf += 1;
            }
        } else {
            *m_buf |= bits << pos_in_word;
            if (len >= 32 - pos_in_word) {
                m_buf += 1;
                *m_buf = bits >> (32 - pos_in_word);
            }
        }
    }

    size_t size() const { return m_size; }

    inline void write_bit(bool val) { write(val, 1); }

    inline void write_unary(uint32_t val) {
        while (val--) {
            write_bit(0);
        }
        write_bit(1);
    }

    inline void write_elias_gamma(uint32_t val) {
        write_unary(utils::bits(val));
        write(val, utils::bits(val));
    }

    inline void write_elias_delta(uint32_t val) {
        write_elias_gamma(utils::bits(val));
        write(val, utils::bits(val));
    }

    inline void write_vbyte(uint32_t val) {
        while (val >= 128) {
            write(0x80 | (val & 0x7f), 8);
            val >>= 7;
        }
        write(0, 1);
        write(val, 7);
    }

   private:
    uint32_t *m_buf;
    size_t    m_size;
};
