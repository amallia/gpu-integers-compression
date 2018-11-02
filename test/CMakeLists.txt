file(GLOB TEST_SOURCES *_test.cpp)
foreach(TEST_SRC ${TEST_SOURCES})
  get_filename_component (TEST_SRC_NAME ${TEST_SRC} NAME_WE)
  add_executable(${TEST_SRC_NAME} ${TEST_SRC})
    target_link_libraries(${TEST_SRC_NAME}
    gtest
    gmock
    )
  add_test(${TEST_SRC_NAME} ${TEST_SRC_NAME})
endforeach(TEST_SRC)

if (CUDA_FOUND)
    CUDA_ADD_EXECUTABLE(cuda_bp_test cuda_bp_test.cu)
    target_link_libraries(cuda_bp_test
    gtest
    gmock
    )

endif()
