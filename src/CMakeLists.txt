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
add_executable(compress_index compress_index.cpp)
target_link_libraries(compress_index
  gpu_integers_compression
  CLI11
)
endif()
