cmake_minimum_required(VERSION 3.0)
project(gpu_integers_compression)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(BUILD_CUDA "Build the CUDA library" ON)

if (NOT CMAKE_BUILD_TYPE)
        message(STATUS "No build type selected, default to Release")
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()
MESSAGE( STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE} )


if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if (CXX_COMPILER_VERSION VERSION_LESS 4.7)
        message(STATUS "GCC version must be at least 4.7!")
    endif()
    set (CMAKE_CXX_FLAGS_RELEASE "-Wall -Wcast-align -Ofast -lm  -DNDEBUG -DHAVE_CXX0X -march=native")
    set (CMAKE_CXX_FLAGS_DEBUG   "-Wall -Wcast-align -ggdb  -lm  -DHAVE_CXX0X -march=native")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    if (CXX_COMPILER_VERSION VERSION_LESS 4.2.1)
        message(STATUS  "Clang version must be at least 4.2.1!" )
    endif()
    set (CMAKE_CXX_FLAGS_RELEASE "-Wall -Wcast-align -O3 -DNDEBUG -DHAVE_CXX0X -msse4.1 -march=native")
    set (CMAKE_CXX_FLAGS_DEBUG   "-Wall -Wcast-align -ggdb  -DHAVE_CXX0X -msse4.1 -march=native")
else ()
    message(FATAL_ERROR "Please, use GCC or Clang compiler!")
endif()

if (USE_SANITIZERS)
 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
endif ()

set(gtest_disable_pthreads ON)


include_directories(include)
add_library(gpu_integers_compression INTERFACE)
target_include_directories(gpu_integers_compression INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_link_libraries(gpu_integers_compression INTERFACE
    mio
    FastPFor
)

if (BUILD_CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDA)
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS "Auto")
    list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
    target_link_libraries(gpu_integers_compression INTERFACE cub)
endif ()
MESSAGE( STATUS "BUILD_CUDA: " ${BUILD_CUDA} )
add_subdirectory(external)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)

enable_testing()
add_subdirectory(bench)


