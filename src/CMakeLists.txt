add_executable(compress_index compress_index.cpp)
target_link_libraries(compress_index
  gpu_integers_compression
  CLI11
)

add_executable(perf_decode perf_decode.cpp)
target_link_libraries(perf_decode
  gpu_integers_compression
  CLI11
)

if(BUILD_CUDA)
CUDA_ADD_EXECUTABLE(cuda_compress_index cuda_compress_index.cu)
target_link_libraries(cuda_compress_index
  gpu_integers_compression
  CLI11
)
CUDA_ADD_EXECUTABLE(cuda_perf_decode cuda_perf_decode.cu)
target_link_libraries(cuda_perf_decode
  gpu_integers_compression
  CLI11
)
endif()
