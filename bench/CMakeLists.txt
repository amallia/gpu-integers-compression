file(GLOB BENCH_SOURCES *_bench.cpp)
foreach(BENCH_SRC ${BENCH_SOURCES})
  get_filename_component (BENCH_SRC_NAME ${BENCH_SRC} NAME_WE)
  add_executable(${BENCH_SRC_NAME} ${BENCH_SRC})
    target_link_libraries(${BENCH_SRC_NAME}
        benchmark
        binary_packing
        synthetic
    )
    target_include_directories(${BENCH_SRC_NAME} BEFORE PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    )
  add_test(${BENCH_SRC_NAME} ${BENCH_SRC_NAME})
endforeach(BENCH_SRC)

if(CUDA_FOUND)
CUDA_ADD_EXECUTABLE(cuda_bp_bench cuda_bp_bench.cu)
  target_link_libraries(cuda_bp_bench
  benchmark
  binary_packing
  synthetic
)
endif()