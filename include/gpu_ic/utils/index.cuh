#pragma once

#include "mappable_vector.hpp"
#include "bit_vector.hpp"

namespace gpu_ic {

    // template <typename BlockCodec>
    class index {
    public:
        index()
            : m_size(0)
        {}

        class builder {
        public:
            builder(uint64_t num_docs)
            {
                m_num_docs = num_docs;
                m_endpoints.push_back(0);
            }

            template <typename DocsIterator, typename Encoder>
            void add_posting_list(uint64_t n, DocsIterator docs_begin, Encoder encoder_function)
            {
                if (!n) throw std::invalid_argument("List must be nonempty");
                tight_variable_byte::encode_single(n, m_lists);
                DocsIterator docs_it(docs_begin);
                std::vector<uint32_t> docs_buf(n);

                uint32_t last_doc(*docs_it++);;
                for (size_t i = 1; i < n; ++i) {
                    uint32_t doc(*docs_it++);
                    docs_buf[i] = doc - last_doc - 1;
                    last_doc = doc;
                }

                std::vector<uint8_t> encoded_values(n*4+1024);
                size_t compressedsize = encoder_function(encoded_values.data(), docs_buf.data(), docs_buf.size());
                encoded_values.resize(compressedsize);
                encoded_values.shrink_to_fit();
                m_lists.insert(m_lists.end(), encoded_values.data(), encoded_values.data() + encoded_values.size());
                m_endpoints.push_back(m_lists.size());
            }


            void build(index& sq)
            {
                sq.m_size = m_endpoints.size() - 1;
                sq.m_num_docs = m_num_docs;
                sq.m_lists.steal(m_lists);
                sq.m_endpoints.steal(m_endpoints);
            }

        private:
            size_t m_num_docs;
            std::vector<uint64_t> m_endpoints;
            std::vector<uint8_t> m_lists;
        };

        size_t size() const
        {
            return m_size;
        }

        uint64_t num_docs() const
        {
            return m_num_docs;
        }


        size_t get_data(std::vector<uint8_t> &data, size_t i) const
        {
            assert(i < size());
            uint32_t n;
            auto data_begin = tight_variable_byte::decode(m_lists.data() + m_endpoints[i], &n, 1);
            data.insert(data.end(), data_begin, m_lists.data() + m_endpoints[i+1] );
            return n;
        }

        void warmup(size_t i) const
        {
            assert(i < size());
        //     compact_elias_fano::enumerator endpoints(m_endpoints, 0,
        //                                              m_lists.size(), m_size,
        //                                              m_params);

            auto begin = m_endpoints[i];
            auto end = m_lists.size();
            if (i + 1 != size()) {
                end = m_endpoints[i + 1];
            }

            volatile uint32_t tmp;
            for (size_t i = begin; i != end; ++i) {
                tmp = m_lists[i];
            }
            (void)tmp;
        }

        void swap(index& other)
        {
            std::swap(m_size, other.m_size);
            m_endpoints.swap(other.m_endpoints);
            m_lists.swap(other.m_lists);
        }

        template <typename Visitor>
        void map(Visitor& visit)
        {
            visit
                (m_size, "m_size")
                (m_num_docs, "m_num_docs")
                (m_endpoints, "m_endpoints")
                (m_lists, "m_lists");
        }

    private:
        size_t m_size;
        size_t m_num_docs;
        mapper::mappable_vector<uint64_t> m_endpoints;
        mapper::mappable_vector<uint8_t> m_lists;
    };
}