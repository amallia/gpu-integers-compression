#include "CLI/CLI.hpp"
#include "../external/FastPFor/headers/codecfactory.h"
#include "gpu_ic/utils/binary_freq_collection.hpp"
#include "gpu_ic/utils/progress.hpp"
#include "gpu_ic/utils/bit_ostream.hpp"
#include "gpu_ic/utils/bit_istream.hpp"
#include "gpu_ic/utils/tight_variable_byte.hpp"
#include "gpu_ic/utils/index.hpp"
#include "gpu_ic/utils/mapper.hpp"
#include "gpu_ic/utils/utils.hpp"
#include "mio/mmap.hpp"
#include <chrono>

using namespace gpu_ic;
using namespace FastPForLib;
using clock_type = std::chrono::high_resolution_clock;

template<typename Codec>
void perftest(const std::string &filename)
{
    gpu_ic::index<Codec> coll;
    mio::mmap_source m(filename);
    mapper::map(coll, m);

    size_t min_length =  500;
    size_t max_length = 5000000;
    size_t max_number_of_lists = 5000;

    std::vector<size_t> long_lists;
    long_lists.reserve(max_number_of_lists);
    std::cout << "warming up " << coll.size() << " posting lists" << std::endl;
    for (size_t i = 0; i < coll.size() and long_lists.size() <= max_number_of_lists; ++i) {
        if (coll[i].size() >= min_length and coll[i].size() < max_length) {
            long_lists.push_back(i);
            coll.warmup(i);
        }
    }
    std::cout << "Scanning " << long_lists.size() << " posting lists, whose length is between " << min_length << " and " << max_length << std::endl;
    auto start = clock_type::now();
    size_t postings = 0;
    for (auto i: long_lists) {
        auto reader = coll[i];
        size_t size = reader.size();
        for (size_t i = 0; i < size; ++i) {
            reader.next();
            utils::do_not_optimize_away(reader.docid());
        }
        postings += size;
    }
    auto end = clock_type::now();
    std::chrono::duration<double> elapsed = end - start;

    double next_ns = elapsed.count() / postings * 1000000000;
    double b_int_s = postings / elapsed.count() / 1000000;
    std::cout << "Performed " << postings << " next()"
             << " in " << elapsed.count() << " [sec], "
             << std::fixed << std::setprecision(2)
             << next_ns << " [ns] x posting, "
             << b_int_s << " M ints/sec"
             << std::endl;

}

bool read_query(std::vector<uint32_t> &ret, std::istream &is = std::cin) {
    ret.clear();
    std::string line;
    if (!std::getline(is, line))
        return false;
    std::istringstream iline(line);
    uint32_t       term_id;
    while (iline >> term_id) {
        ret.push_back(term_id);
    }
    return true;
}

int main(int argc, char const *argv[])
{
    std::string type;
    std::string index_basename;
    // std::string query_basename;

    CLI::App app{"compress_index - a tool for compressing an index."};
    app.add_option("-t,--type", type, "Index type")->required();
    app.add_option("-i,--index", index_basename, "Index basename")->required();
    // app.add_option("-q,--query", query_basename, "Query basename")->required();
    CLI11_PARSE(app, argc, argv);

    // std::vector<uint32_t> terms;
    // std::filebuf fb;
    // size_t queries_num = 0;
    // if (fb.open(query_basename, std::ios::in)) {
    //     std::istream is(&fb);
    //     std::vector<uint32_t> q;
    //     while (read_query(q, is)) {
    //         queries_num+=1;
    //         terms.insert(terms.end(), q.begin(), q.end());
    //     }
    // }

    if (type == "simdbp") {
        CompositeCodec<SIMDBinaryPacking, VariableByte> codec;
        perftest<decltype(codec)>(index_basename);
    } else if (type == "streamvbyte") {
        StreamVByte codec;
        perftest<decltype(codec)>(index_basename);
    } else if (type == "bp") {
        CompositeCodec<BP32, VariableByte> codec;
        perftest<decltype(codec)>(index_basename);
    } else if (type == "varintgb") {
        VarIntGB<> codec;
        perftest<decltype(codec)>(index_basename);
    } else {
        std::cerr << "Unknown type" << std::endl;
    }



    // std::cout << "Queries: " << queries_num << ", terms: " << terms.size() << std::endl;


    return 0;
}
