#include "CLI/CLI.hpp"
#include "gpu_ic/cuda_bp.cuh"
#include "gpu_ic/cuda_vbyte.cuh"

#include "gpu_ic/utils/binary_freq_collection.hpp"
#include "gpu_ic/utils/progress.hpp"
#include "gpu_ic/utils/tight_variable_byte.hpp"
#include "gpu_ic/utils/index.cuh"
#include "gpu_ic/utils/mapper.hpp"
#include "gpu_ic/utils/utils.hpp"
#include "mio/mmap.hpp"
#include <chrono>
#include <cmath>

using namespace gpu_ic;
using clock_type = std::chrono::high_resolution_clock;

template <typename Decoder>
void perftest(const std::string &filename, Decoder &decoder_function, const std::vector<uint32_t> &terms)
{
    gpu_ic::index coll;
    mio::mmap_source m;
    std::error_code error;
    m.map(filename, error);
    mapper::map(coll, m);

    std::vector<std::pair<size_t, std::vector<uint8_t>>> long_lists;
    long_lists.reserve(terms.size());
    for(auto&& t :terms) {
	std::vector<uint8_t> tmp;
        auto n = coll.get_data(tmp, t);
        long_lists.push_back(std::make_pair(n, tmp));
    }
    CUDA_CHECK_ERROR(cudaSetDevice(0));
    warmUpGPU<<<1, 1>>>();
    std::cout << "Scanning " << long_lists.size() << " posting lists, whose length is between " << min_length << std::endl;
    std::chrono::duration<double> elapsed(0);
    size_t postings = 0;
    for (auto i: long_lists) {
        uint8_t *  d_encoded;
        CUDA_CHECK_ERROR(cudaMalloc((void **)&d_encoded, i.second.size() * sizeof(uint8_t)));
        CUDA_CHECK_ERROR(cudaMemcpy(d_encoded, i.second.data(), i.second.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));

        std::vector<uint32_t> decode_values(i.first);
        uint32_t * d_decoded;
        CUDA_CHECK_ERROR(cudaMalloc((void **)&d_decoded, decode_values.size() * sizeof(uint32_t)));
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        auto start = clock_type::now();
        decoder_function(d_decoded, d_encoded, decode_values.size());
        cudaDeviceSynchronize();
        auto end = clock_type::now();

        CUDA_CHECK_ERROR(cudaMemcpy(decode_values.data(), d_decoded, decode_values.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        CUDA_CHECK_ERROR(cudaFree(d_encoded));
        CUDA_CHECK_ERROR(cudaFree(d_decoded));

    	elapsed += end - start;

        for (size_t j = 0; j < i.first; ++j) {
            // std::cerr << decode_values[j] << std::endl;
            utils::do_not_optimize_away(decode_values[j]);
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
    std::string query_basename;

    CLI::App app{"compress_index - a tool for compressing an index."};
    app.add_option("-t,--type", type, "Index type")->required();
    app.add_option("-i,--index", index_basename, "Index basename")->required();
    app.add_option("-q,--query", query_basename, "Query basename")->required();
    CLI11_PARSE(app, argc, argv);

     std::vector<uint32_t> terms;
     std::filebuf fb;
     size_t queries_num = 0;
     if (fb.open(query_basename, std::ios::in)) {
         std::istream is(&fb);
         std::vector<uint32_t> q;
         while (read_query(q, is)) {
             queries_num+=1;
             terms.insert(terms.end(), q.begin(), q.end());
         }
     }
    if (type == "cuda_bp") {
        perftest(index_basename, cuda_bp::decode, terms);
    } else if (type == "cuda_vbyte") {
        perftest(index_basename, cuda_vbyte::decode<>, terms);
    } else {
        std::cerr << "Unknown type" << std::endl;
    }

    // std::cout << "Queries: " << queries_num << ", terms: " << terms.size() << std::endl;


    return 0;
}
