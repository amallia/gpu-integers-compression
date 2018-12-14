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

class bit_istream {
   public:
    bit_istream(uint8_t const *in)
        : m_in(reinterpret_cast<const uint32_t *>(in)), m_avail(0), m_buf(0), m_pos(0) {}

    size_t position() const { return m_pos; }

    uint32_t read(uint32_t len) {
        if (!len)
            return 0;

        if (m_avail < len) {
            m_buf |= uint64_t(*m_in++) << m_avail;
            m_avail += 32;
        }
        uint32_t val = m_buf & ((uint64_t(1) << len) - 1);
        m_buf >>= len;
        m_avail -= len;
        m_pos += len;

        return val;
    }

    inline uint8_t read_bit() { return read(1); }

    inline uint32_t read_unary() {
        uint32_t v = 0;
        while (read_bit() == 0)
            ++v;
        return v;
    }

    inline uint32_t read_elias_gamma() {
        auto bits = read_unary();
        return read(bits);
    }

    inline uint32_t read_elias_delta() {
        auto bits = read_elias_gamma();
        return read(bits);
    }

    inline uint32_t read_vbyte() {
        uint32_t val = 0;
        size_t   i   = 0;
        while (read_bit()) {
            val |= read(7) << (7 * i++);
            ;
        }
        val |= read(7) << (7 * i);
        return val;
    }

   private:
    uint32_t const *m_in;
    uint32_t        m_avail;
    uint64_t        m_buf;
    size_t          m_pos;
};