#include "CLI/CLI.hpp"
#include "../external/FastPFor/headers/codecfactory.h"
#include "gpu_ic/utils/binary_freq_collection.hpp"
#include "gpu_ic/utils/progress.hpp"
#include "gpu_ic/utils/bit_ostream.hpp"
#include "gpu_ic/utils/bit_istream.hpp"
#include "gpu_ic/utils/tight_variable_byte.hpp"
#include "gpu_ic/utils/index.hpp"
#include "gpu_ic/utils/mapper.hpp"
#include "mio/mmap.hpp"

using namespace gpu_ic;
using namespace FastPForLib;

template <typename InputCollection, typename Codec>
void verify_index(InputCollection const &input,
                       const std::string &filename, bool compress_freqs) {

    Codec codec;
    gpu_ic::index<Codec> coll;
    mio::mmap_source m;
    std::error_code error;
    m.map(filename, error);
    mapper::map(coll, m);

    {
        progress progress("Verify index", input.size());

        size_t i =0;
        for (auto const &plist : input) {
            auto docs_it = compress_freqs ?  plist.freqs.begin() : plist.docs.begin();

            std::vector<uint32_t> values(plist.docs.size());
            uint32_t last_doc(*docs_it++);;
            for (size_t j = 1; j < plist.docs.size(); ++j) {
                uint32_t doc(*docs_it++);
                if(not compress_freqs){
                    values[j] = doc - last_doc - 1;
                }
                else{
                    values[j] = doc - 1;
                }
                last_doc = doc;
            }

            std::vector<uint8_t> tmp;
            auto n = coll.get_data(tmp, i);
            std::vector<uint32_t> decode_values(n);
            codec.decodeArray(reinterpret_cast<uint32_t const *>(tmp.data()), tmp.size()/4, reinterpret_cast<uint32_t*>(decode_values.data()), n);

            if(n != plist.docs.size())
            {
                std::cerr << "Error: wrong list length. List: " << i << ", size: " << n << ", real_size: " << plist.docs.size() << std::endl;
                std::abort();
            }

            for (size_t j = 0; j < n; ++j) {
                if(decode_values[j] != values[j]) {
                    std::cerr << "Error: wrong decoded value. List: " << i << ", position: " << j << ", element: " << decode_values[j] << ", real_element: " << values[j] << std::endl;
                    std::abort();
                }
            }
            progress.update(1);
            i+=1;
        }
    }

}

template <typename InputCollection, typename Codec>
void create_collection(InputCollection const &input,
                       const std::string &output_filename,
                       Codec &codec, bool compress_freqs) {

    typename gpu_ic::index<Codec>::builder builder(input.num_docs());
    size_t postings = 0;
    {
        progress progress("Create index", input.size());

        for (auto const &plist : input) {
            size_t size = plist.docs.size();
            if(not compress_freqs) {
                builder.add_posting_list(size, plist.docs.begin(), codec, compress_freqs);
            }
            else {
                builder.add_posting_list(size, plist.freqs.begin(), codec, compress_freqs);
            }
            postings += size;
            progress.update(1);
        }
    }

    gpu_ic::index<Codec> coll;
    auto data_len = builder.build(coll);
    auto byte= mapper::freeze(coll, output_filename.c_str());


    double bits_per_doc  = data_len * 8.0 / postings;
    std::cout << "Documents: " << postings << ", Total size bytes: " << byte << ", bits/doc: " << bits_per_doc << std::endl;

    verify_index<InputCollection, Codec>(input, output_filename, compress_freqs);
}


int main(int argc, char const *argv[])
{
    std::string type;
    std::string input_basename;
    std::string output_filename;
    bool compress_freqs = false;

    CLI::App app{"compress_index - a tool for compressing an index."};
    app.add_option("-t,--type", type, "Index type")->required();
    app.add_option("-c,--collection", input_basename, "Collection basename")->required();
    app.add_option("-o,--output", output_filename, "Output filename")->required();
    app.add_flag("--freqs", compress_freqs, "Compress freqs instead of docs");

    CLI11_PARSE(app, argc, argv);

    binary_freq_collection input(input_basename.c_str());
    if (type == "simdbp") {
        CompositeCodec<SIMDBinaryPacking, VariableByte> codec;
        create_collection(input, output_filename, codec, compress_freqs);
    } else if (type == "streamvbyte") {
        StreamVByte codec;
        create_collection(input, output_filename, codec, compress_freqs);
    } else if (type == "bp") {
        CompositeCodec<BP32, VariableByte> codec;
        create_collection(input, output_filename, codec, compress_freqs);
    } else if (type == "varintgb") {
        VarIntGB<> codec;
        create_collection(input, output_filename, codec, compress_freqs);
    } else {
        std::cerr << "Unknown type" << std::endl;
    }

    return 0;
}
