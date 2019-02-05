#pragma once

#include "tight_variable_byte.hpp"

namespace gpu_ic {

    struct posting_list {

        template <typename DocsIterator, typename Codec>
        static void write(std::vector<uint8_t>& out, uint32_t n, DocsIterator docs_begin, Codec codec, bool compress_freqs) {
            tight_variable_byte::encode_single(n, out);

            DocsIterator docs_it(docs_begin);
            std::vector<uint32_t> docs_buf(n);

            uint32_t last_doc(*docs_it++);;
            for (size_t i = 1; i < n; ++i) {
                uint32_t doc(*docs_it++);
                if(not compress_freqs) {
                    docs_buf[i] = doc - last_doc - 1;
                }
                else {
                    docs_buf[i] = doc - 1;
                }
                last_doc = doc;
            }

            size_t compressedsize = 0;
            std::vector<uint8_t> encoded_values(n*4+1024);
            codec.encodeArray(docs_buf.data(), n, reinterpret_cast<uint32_t*>(encoded_values.data()), compressedsize);
            out.insert(out.end(), encoded_values.data(), encoded_values.data() + compressedsize*4);
        }

        class document_enumerator {
        public:

            template <typename Codec>
            document_enumerator(uint8_t const* data, uint64_t len, Codec codec)
                : m_n(0)
                , m_base(tight_variable_byte::decode(data, &m_n, 1))
                , m_len(len)
            {
                m_docs_buf.resize(m_n);
                decode_docs_block(codec);

            }

            void next()
            {
                ++m_pos_in_block;
                m_cur_docid = m_docs_buf[m_pos_in_block];
            }


            uint64_t docid() const
            {
                return m_cur_docid;
            }


            uint64_t position() const
            {
                return m_pos_in_block;
            }

            uint64_t size() const
            {
                return m_n;
            }

        private:

            template <typename Codec>
            void decode_docs_block(Codec codec)
            {

                size_t n =m_n;
                codec.decodeArray(reinterpret_cast<uint32_t const *>(m_base), m_len/4, reinterpret_cast<uint32_t*>(m_docs_buf.data()), n);

                m_pos_in_block = 0;
                m_cur_docid = m_docs_buf[0];

            }

            uint32_t m_n;
            uint8_t const* m_base;
            uint64_t m_len;

            uint32_t m_pos_in_block;
            uint32_t m_cur_docid;


            std::vector<uint32_t> m_docs_buf;

        };

    };
}