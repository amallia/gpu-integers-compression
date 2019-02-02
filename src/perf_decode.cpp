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
#include <cmath>

using namespace gpu_ic;
using namespace FastPForLib;
using clock_type = std::chrono::high_resolution_clock;

template<typename Codec>
void perftest(const std::string &filename)
{
    Codec codec;
    gpu_ic::index<Codec> coll;
    mio::mmap_source m;
    std::error_code error;
    m.map(filename, error);
    mapper::map(coll, m);

    size_t min_length = pow(2, 15); ;
    size_t max_number_of_lists = 500000000;

    std::vector<std::pair<size_t, std::vector<uint8_t>>> long_lists;
    long_lists.reserve(max_number_of_lists);
    for (size_t i = 0; i < coll.size() and long_lists.size() <= max_number_of_lists; ++i) {
        std::vector<uint8_t> tmp;
        auto n = coll.get_data(tmp, i);
        if (n >= min_length) {
            long_lists.push_back(std::make_pair(n, tmp));
        }
    }
    std::cout << "Scanning " << long_lists.size() << " posting lists, whose length is between " << min_length << std::endl;
    std::chrono::duration<double> elapsed(0);
    size_t postings = 0;
    for (auto i: long_lists) {
        std::vector<uint32_t> decode_values(i.first);
    	auto start = clock_type::now();
        size_t n = 0;
        codec.decodeArray(reinterpret_cast<uint32_t const *>(i.second.data()), i.second.size()/4, reinterpret_cast<uint32_t*>(decode_values.data()), n);
    	auto end = clock_type::now();
    	elapsed += end - start;
        if(n != i.first) {
            std::cerr << "Error: number of decoded values " << n << ", actual number of values" << i.first << std::endl;
        }
        for (size_t i = 0; i < n; ++i) {
            utils::do_not_optimize_away(decode_values[i]);
        }
        postings += decode_values.size();
    }

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
