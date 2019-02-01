#include "CLI/CLI.hpp"
#include "../external/FastPFor/headers/codecfactory.h"
#include "gpu_ic/utils/binary_freq_collection.hpp"
#include "gpu_ic/utils/progress.hpp"
#include "gpu_ic/utils/bit_ostream.hpp"
#include "gpu_ic/utils/bit_istream.hpp"
#include "gpu_ic/utils/tight_variable_byte.hpp"
#include "gpu_ic/utils/index.hpp"
#include "gpu_ic/utils/mapper.hpp"

using namespace gpu_ic;
using namespace FastPForLib;

// template <typename InputCollection, typename Codec>
// void verify_index(InputCollection const &input,
//                        const std::string &output_filename,
//                        Codec &codec) {

//     std::ifstream fin(output_filename, std::ios::binary);

//     uint64_t endpoints_size;
//     fin.read((char*)&endpoints_size, 8);
//     std::vector<uint32_t> endpoints(endpoints_size);
//     fin.read((char*)endpoints.data(), endpoints_size);

//     uint64_t docs_size;
//     fin.read((char*)&docs_size, 8);
//     std::vector<uint8_t> payload(docs_size);
//     fin.read((char*)payload.data(), docs_size);

//     {
//         progress progress("Create index", input.size());

//         size_t i =0;
//         for (auto const &plist : input) {
//             size_t start = endpoints[i];
//             size_t len = endpoints[i+1] - start;

//             uint32_t n;
//             const uint8_t * p = tight_variable_byte::decode(payload.data()+start, &n, 1);

//             std::cerr << start << std::endl;
//             if(n != plist.docs.size())
//             {
//                 std::cerr << "Error: wrong list length. List: " << i << ", size: " << n << ", real_size: " << plist.docs.size() << std::endl;
//                 std::abort();
//             }

//             auto docs_it = plist.docs.begin();

//             std::vector<uint32_t> values(plist.docs.size());
//             uint32_t last_doc = 0;
//             for (size_t j = 0; j < plist.docs.size(); ++j) {
//                 uint32_t doc(*docs_it++);
//                 values[j] = doc - last_doc - 1;
//                 last_doc = doc;
//             }
//             std::vector<uint32_t> decoded_values(n);
//             // size_t nn = n;
//             // codec.decodeArray(reinterpret_cast<uint32_t const *>(p), len, reinterpret_cast<uint32_t*>(decoded_values.data()), nn);

//             // for (int j = 0; j < plist.docs.size(); ++j)
//             // {
//             //     if(values[j] != decoded_values[j]) {
//             //         std::cerr << "Error: wrong decoded value. List: " << i << ", position: " << j << ", element: " << decoded_values[j] << ", real_element: " << values[j] << std::endl;
//             //         std::abort();
//             //     }
//             // }

//             progress.update(1);
//             i+=1;
//         }
//     }

// }

template <typename InputCollection, typename Codec>
void create_collection(InputCollection const &input,
                       const std::string &output_filename,
                       Codec &codec) {

    gpu_ic::index::builder builder(input.num_docs());
    size_t postings = 0;
    {
        progress progress("Create index", input.size());

        for (auto const &plist : input) {
            size_t size = plist.docs.size();
            builder.add_posting_list(size, plist.docs.begin(), codec);
            postings += size;
            progress.update(1);
        }
    }

    gpu_ic::index coll;
    builder.build(coll);
    auto byte= mapper::freeze(coll, output_filename.c_str());


    double bits_per_doc  = byte * 8.0 / postings;
    std::cout << "Documents: " << postings << ", bytes: " << byte << ", bits/doc: " << bits_per_doc << std::endl;

    // verify_index(input, output_filename, codec);
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
    if (type == "simdbp") {
        CompositeCodec<SIMDBinaryPacking, VariableByte> codec;
        create_collection(input, output_filename, codec);
    } else if (type == "streamvbyte") {
        StreamVByte codec;
        create_collection(input, output_filename, codec);
    } else if (type == "bp") {
        CompositeCodec<BP32, VariableByte> codec;
        create_collection(input, output_filename, codec);
    } else if (type == "varintgb") {
        VarIntGB<> codec;
        create_collection(input, output_filename, codec);
    } else {
        std::cerr << "Unknown type" << std::endl;
    }

    return 0;
}