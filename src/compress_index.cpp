#include "CLI/CLI.hpp"
// #include "gpu_ic/cuda_bp.cuh"
#include "../external/FastPFor/headers/codecfactory.h"
#include "gpu_ic/utils/binary_freq_collection.hpp"

using namespace gpu_ic;
using namespace FastPForLib;
template <typename InputCollection, typename Codec>
void create_collection(InputCollection const &input,
                       const std::string &output_filename,
                       Codec &codec) {
    std::ofstream fout(output_filename, std::ios::binary);

    std::vector<uint8_t> payload;
    std::vector<uint32_t> endpoints;
    endpoints.push_back(0);

    size_t postings = 0;
    for (auto const &plist : input) {


        size_t size = plist.docs.size();
        std::vector<uint32_t> values(size);
        std::vector<uint8_t> encoded_values(size*4);

        auto docs_it = plist.docs.begin();

        uint32_t last_doc = 0;
        for (size_t i = 0; i < size; ++i) {
            uint32_t doc(*docs_it++);
            values[i] = doc - last_doc - 1;
            last_doc = doc;
        }
        size_t compressedsize = 0;
        codec.encodeArray(values.data(), values.size(), reinterpret_cast<uint32_t*>(encoded_values.data()), compressedsize);
        payload.insert(payload.end(), encoded_values.data(), encoded_values.data() + compressedsize*4);

        postings += size;
        endpoints.push_back(encoded_values.size());


    }

    size_t docs_size = payload.size();
    double bits_per_doc  = docs_size * 8.0 / postings;
    std::cout << "Documents: " << postings << ", bytes: " << docs_size << ", bits/doc: " << bits_per_doc << std::endl;
}

int main(int argc, char const *argv[])
{
    std::string type;
    std::string input_basename;
    std::string output_filename;

    CLI::App app{"compress_index - a tool for compressing an index."};
    app.add_option("-t,--type", type, "Index type")->required();
    app.add_option("-c,--collection", input_basename, "Collection basename")->required();
    app.add_option("-o,--output", output_filename, "Output filename")->required();
    CLI11_PARSE(app, argc, argv);

    binary_freq_collection input(input_basename.c_str());
    if (type == "cuda_bp") {
        // create_collection<binary_freq_collection, cuda_bp>(input, output_filename, type);
    } else if (type == "cuda_vbyte") {
    } else if (type == "simdbp") {
    } else if (type == "streamvbyte") {
        IntegerCODEC &codec = *CODECFactory::getFromName("streamvbyte");
        create_collection(input, output_filename, codec);
    } else if (type == "bp") {
    } else if (type == "varintgb") {
    } else {
        std::cerr << "Unknown type" << std::endl;
    }

    return 0;
}