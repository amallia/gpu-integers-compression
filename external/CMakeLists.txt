EXECUTE_PROCESS(COMMAND git submodule update --init
                WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
                OUTPUT_QUIET
        )

# Add FastPFor
add_subdirectory(FastPFor EXCLUDE_FROM_ALL)

add_subdirectory(googletest EXCLUDE_FROM_ALL)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)
add_subdirectory(benchmark EXCLUDE_FROM_ALL)

add_library(cub INTERFACE)
target_include_directories(cub INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/cub>
)