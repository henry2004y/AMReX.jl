cmake_minimum_required(VERSION 3.18)
project(amrex_jl_shim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenMP REQUIRED)

add_library(amrex_jl_shim SHARED amrex_jl_shim.cpp)

target_include_directories(amrex_jl_shim PRIVATE
    ${JULIA_INCLUDE_DIR}
    ${CXXWRAP_INCLUDE_DIR}
    ${AMREX_INCLUDE_DIR}
    ${EXTRA_INCLUDE_DIRS}
)

target_link_directories(amrex_jl_shim PRIVATE
    ${JULIA_LIB_DIR}
    ${CXXWRAP_LIB_DIR}
    ${AMREX_LIB_DIR}
    ${EXTRA_LIB_DIRS}
)

target_link_libraries(amrex_jl_shim PRIVATE
    OpenMP::OpenMP_CXX
    amrex_3d
    cxxwrap_julia
    julia
)

set_target_properties(amrex_jl_shim PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${AMREX_LIB_DIR};${CXXWRAP_LIB_DIR};${EXTRA_RPATH_DIRS}"
)

