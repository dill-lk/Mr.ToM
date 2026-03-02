cmake_minimum_required(VERSION 3.18)
project(RMoEClinicalEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(rmoe_engine
    src/main.cpp
    src/DiagnosticEngine.cpp
)

target_include_directories(rmoe_engine PRIVATE include)

target_compile_options(rmoe_engine PRIVATE
    -O3
    -mavx2
    -mfma
    -ffast-math
)

find_package(OpenMP REQUIRED)
target_link_libraries(rmoe_engine PRIVATE OpenMP::OpenMP_CXX)

find_package(nlohmann_json QUIET)
if(nlohmann_json_FOUND)
    target_link_libraries(rmoe_engine PRIVATE nlohmann_json::nlohmann_json)
endif()

# llama.cpp backend integration (expects llama.cpp as a subdirectory)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/CMakeLists.txt")
    add_subdirectory(llama.cpp)
    target_include_directories(rmoe_engine PRIVATE llama.cpp/include)

    if(TARGET llama)
        target_link_libraries(rmoe_engine PRIVATE llama)
        target_compile_definitions(rmoe_engine PRIVATE RMOE_HAS_LLAMA=1)
    endif()

    if(TARGET common)
        target_link_libraries(rmoe_engine PRIVATE common)
    endif()
endif()
