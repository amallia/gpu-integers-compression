#include "CLI/CLI.hpp"
#include "gpu_ic/utils/binary_freq_collection.hpp"
#include "gpu_ic/utils/progress.hpp"
#include "gpu_ic/cuda_bp.cuh"
#include "gpu_ic/cuda_vbyte.cuh"
#include "gpu_ic/utils/cuda_utils.hpp"

using namespace gpu_ic;

template <typename InputCollection, typename Codec>
void verify_index(InputCollection const &input,
                       const std::string &filename) {


    mio::mmap_source m;
    std::error_code error;
    m.map(filename, error);
    uint32_t endpoints_size;
    std::copy(m.data(), m.data() + 4, &endpoints_size);
    std::vector<uint64_t> endpoints(endpoints_size);
    std::copy(m.data() + 4, m.data() + 4 + 8*endpoints_size,  endpoints.data());
    auto payload = m.data() + 4 + 8*endpoints_size;
    {
        progress progress("Verify index", input.size());

        size_t i =0;
        for (auto const &plist : input) {
            auto docs_it = plist.docs.begin();

            std::vector<uint32_t> values(plist.docs.size());
            uint32_t last_doc(*docs_it++);;
            for (size_t j = 1; j < plist.docs.size(); ++j) {
                uint32_t doc(*docs_it++);
                values[j] = doc - last_doc - 1;
                last_doc = doc;
            }
            auto len = endpoints[i+1] - endpoints[i];
            uint8_t *  d_encoded;
            CUDA_CHECK_ERROR(cudaMalloc((void **)&d_encoded, len * sizeof(uint8_t)));

            size_t n;
            auto data_begin = tight_variable_byte::decode(payload + endpoints[i], &n, 1);
            CUDA_CHECK_ERROR(cudaMemcpy(d_encoded, data_begin, len * sizeof(uint8_t), cudaMemcpyHostToDevice));

            std::vector<uint32_t> decode_values(n);
            uint32_t * d_decoded;
            CUDA_CHECK_ERROR(cudaMalloc((void **)&d_decoded, values.size() * sizeof(uint32_t)));
            Encoder::decode(d_decoded, d_encoded, decode_values.size());
            CUDA_CHECK_ERROR(cudaMemcpy(decode_values.data(), d_decoded, values.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

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



template <typename InputCollection, typename Encoder>
void create_collection(InputCollection const &input,
                       const std::string &output_filename,
		       const Encoder &encode_function) {
    std::ofstream fout(output_filename, std::ios::binary);

    std::vector<uint8_t> payload;
    std::vector<uint64_t> endpoints;
    endpoints.push_back(0);

    size_t postings = 0;
    {
        pisa::progress progress("Create index", input.size());

        for (auto const &plist : input) {
            size_t size = plist.docs.size();

            std::vector<uint8_t> len(5);
            bit_ostream bw(len.data());
            bw.write_vbyte(size);
            payload.insert(payload.end(), len.data(), len.data() + bw.size()/8);

	        std::vector<uint32_t> values(size);
            std::vector<uint8_t> encoded_values(size*4+1024);

            auto docs_it = plist.docs.begin();

            uint32_t last_doc = 0;
            for (size_t i = 0; i < size; ++i) {
                uint32_t doc(*docs_it++);
                values[i] = doc - last_doc - 1;
                last_doc = doc;
            }
            auto compressedsize = encode_function(encoded_values.data(), values.data(), values.size());
            payload.insert(payload.end(), encoded_values.data(), encoded_values.data() + compressedsize);
            postings += size;
            endpoints.push_back(encoded_values.size());
            progress.update(1);
        }
    }
    uint32_t endpoints_size = endpoints.size();
    fout.write((const char*)&endpoints_size, 4);
    fout.write((const char*)endpoints.data(), endpoints_size);

    size_t docs_size = payload.size();
    fout.write((const char*)&docs_size, 4);
    fout.write((const char*)payload.data(), docs_size);

    double bits_per_doc  = fout.tellp()*8.0 / postings;
    std::cout << "Documents: " << postings << ", bytes: " << fout.tellp() << ", bits/doc: " << bits_per_doc << std::endl;
    verify_index<InputCollection, Encoder>(input, output_filename);

}

int main(int argc, char** argv) {
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
        create_collection(input, output_filename, cuda_bp::encode<>);
    } else if (type == "cuda_vbyte") {
        create_collection(input, output_filename, cuda_vbyte::encode<>);
    } else {
        std::cerr << "Unknown type" << std::endl;
    }
}

}
