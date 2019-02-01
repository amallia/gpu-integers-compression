#pragma once

#include "mappable_vector.hpp"
#include "bit_vector.hpp"

#include "posting_list.hpp"

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

            template <typename DocsIterator, typename Codec>
            void add_posting_list(uint64_t n, DocsIterator docs_begin, Codec codec)
            {
                if (!n) throw std::invalid_argument("List must be nonempty");
                posting_list::write(m_lists, n, docs_begin, codec);
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

        // typedef typename block_posting_list<BlockCodec, Profile>::document_enumerator document_enumerator;

        // document_enumerator operator[](size_t i) const
        // {
        //     assert(i < size());
        //     // compact_elias_fano::enumerator endpoints(m_endpoints, 0,
        //     //                                          m_lists.size(), m_size,
        //     //                                          m_params);

        //     // auto endpoint = endpoints.move(i).second;
        //     auto endpoint = m_endpoints[i];
        //     return document_enumerator(m_lists.data() + endpoint, num_docs(), i);
        // }

        // void warmup(size_t i) const
        // {
        //     assert(i < size());
        //     compact_elias_fano::enumerator endpoints(m_endpoints, 0,
        //                                              m_lists.size(), m_size,
        //                                              m_params);

        //     auto begin = endpoints.move(i).second;
        //     auto end = m_lists.size();
        //     if (i + 1 != size()) {
        //         end = endpoints.move(i + 1).second;
        //     }

        //     volatile uint32_t tmp;
        //     for (size_t i = begin; i != end; ++i) {
        //         tmp = m_lists[i];
        //     }
        //     (void)tmp;
        // }

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
                (m_lists, "m_lists")
                ;
        }

    private:
        size_t m_size;
        size_t m_num_docs;
        // bit_vector m_endpoints;
        mapper::mappable_vector<uint64_t> m_endpoints;
        mapper::mappable_vector<uint8_t> m_lists;
    };
}