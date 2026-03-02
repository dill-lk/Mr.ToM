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
    # common utils are required by the mtmd CLI tools bundled with llama.cpp
    set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build llama.cpp common utils (required for mtmd)" FORCE)
    add_subdirectory(llama.cpp)
    target_include_directories(rmoe_engine PRIVATE llama.cpp/include)

    if(TARGET llama)
        target_link_libraries(rmoe_engine PRIVATE llama)
        target_compile_definitions(rmoe_engine PRIVATE RMOE_HAS_LLAMA=1)
    endif()

    if(TARGET common)
        target_link_libraries(rmoe_engine PRIVATE common)
    endif()

    # mtmd: multimodal (CLIP + LLM vision) support
    # The CLIP mmproj model (vision_proj.gguf) is loaded via mtmd_init_from_file,
    # NOT via llama_model_load_from_file which rejects CLIP architectures.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd/CMakeLists.txt" AND TARGET llama)
        # LLAMA_INSTALL_VERSION is set inside the llama.cpp sub-scope and is not
        # propagated to our root scope, so provide a fallback before adding mtmd.
        if(NOT DEFINED LLAMA_INSTALL_VERSION)
            set(LLAMA_INSTALL_VERSION "0.0.0")
        endif()
        add_subdirectory(llama.cpp/tools/mtmd)
        if(TARGET mtmd)
            target_link_libraries(rmoe_engine PRIVATE mtmd)
            target_include_directories(rmoe_engine PRIVATE llama.cpp/tools/mtmd)
            target_compile_definitions(rmoe_engine PRIVATE RMOE_HAS_MTMD=1)
        endif()
    endif()
endif()
