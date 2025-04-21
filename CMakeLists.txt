cmake_minimum_required(VERSION 3.18)

# Project configuration: enable both C++ and CUDA languages.
project(RayTracingInOneWeekend LANGUAGES CXX CUDA)

# Enable folder grouping in IDEs like Visual Studio.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT RayTracingInOneWeekend)

# Collect source files. Adjust paths as needed (this example assumes your sources are in the project root and subfolders).
file(GLOB_RECURSE SRC_FILES
  "${CMAKE_SOURCE_DIR}/src/*.h"
  "${CMAKE_SOURCE_DIR}/src/*.hpp"

  "${CMAKE_SOURCE_DIR}/src/*.cpp"
  "${CMAKE_SOURCE_DIR}/src/*.cu"
  "${CMAKE_SOURCE_DIR}/src/*.c"
)

set(GLM_BUILD_LIBRARY OFF)
add_subdirectory(Vendor/glm)
add_subdirectory(Vendor/spdlog)
add_subdirectory(Vendor/SFML-3.0.0)

# Disable undesired SFML modules
set(SFML_BUILD_AUDIO OFF CACHE BOOL "Disable SFML Audio module" FORCE)
set(SFML_BUILD_NETWORK OFF CACHE BOOL "Disable SFML Network module" FORCE)

if(TARGET sfml-main)
  # Exclude the sfml-main target from the ALL build so it doesn't get built by default.
  set_target_properties(sfml-main PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

set(GLM_ENABLE_FAST_MATH ON CACHE BOOL "Enable fast math" FORCE)
set(GLM_ENABLE_CXX_20 ON CACHE BOOL "Enable C++20" FORCE)

# Set vendor target folders.
set_property(TARGET spdlog PROPERTY FOLDER "Vendor/spdlog")
set_property(TARGET glm PROPERTY FOLDER "Vendor/glm")
set_property(TARGET sfml-system PROPERTY FOLDER "Vendor/SFML")
set_property(TARGET sfml-window PROPERTY FOLDER "Vendor/SFML")
set_property(TARGET sfml-graphics PROPERTY FOLDER "Vendor/SFML")
set_property(TARGET sfml-main PROPERTY FOLDER "Vendor/SFML")
set_property(TARGET freetype PROPERTY FOLDER "Vendor/SFML") # required by SFML-Graphics

find_package(CUDAToolkit REQUIRED)

# Create the executable.
add_executable(${PROJECT_NAME} ${SRC_FILES})

target_link_libraries(${PROJECT_NAME}
  PRIVATE
  spdlog glm sfml-system sfml-window sfml-graphics
)

# Add option to enable sanitizer for all configurations
option(USE_COMPUTE_SANITIZER "Use CUDA Compute Sanitizer for all configurations" OFF)

# Apply sanitizer if enabled
if(MSVC AND USE_COMPUTE_SANITIZER)
  message(STATUS "Enabling CUDA Compute Sanitizer for all configurations")
  set_target_properties(${PROJECT_NAME} PROPERTIES
    VS_DEBUGGER_COMMAND "${COMPUTE_SANITIZER_PATH}"
    VS_DEBUGGER_COMMAND_ARGUMENTS "--tool memcheck --report-api-errors all --target-processes all \"$<TARGET_FILE:${PROJECT_NAME}>\""
  )
endif()

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Vendor/glm)

target_compile_definitions(${PROJECT_NAME} PRIVATE
  _CONSOLE
  FMT_UNICODE=0

  # GLM-related defines:
  GLM_FORCE_CUDA=1
  GLM_ENABLE_EXPERIMENTAL=1
  GLM_FORCE_CTOR_INIT=1
  GLM_FORCE_XYZW_ONLY=1
  GLM_FORCE_ALIGNED_GENTYPES=1

  # Configuration-specific definitions:
  $<$<CONFIG:Debug>:_DEBUG>
  $<$<CONFIG:Release>:NDEBUG>
)

# For Release builds, enable interprocedural (link time) optimization.
if(CMAKE_BUILD_TYPE MATCHES $<OR:<Release,RelWithDebInfo>>)
  set_target_properties(${PROJECT_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Link with CUDA runtime library (static) similar to linking cudart_static.lib.
# Note: Modern CMake’s FindCUDA support will usually handle linking appropriately,
# but explicitly linking cudart_static is sometimes necessary.
find_library(CUDART_STATIC_LIBRARY cudart_static
  PATHS
  "$ENV{CUDA_PATH}/lib/x64"
  "${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}"
  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/*/lib/x64" # Wildcard for any CUDA version
  PATH_SUFFIXES
  lib64
  lib/x64
  NO_DEFAULT_PATH
)

if(CUDART_STATIC_LIBRARY)
  target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDART_STATIC_LIBRARY})
else()
  message(WARNING "cudart_static library not found! Try to set CUDA_PATH in your environment.")
endif()

# Set the CUDA architecture
set_target_properties(${PROJECT_NAME} PROPERTIES
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_CPPLINT ON
  CUDA_CPPCHECK ON

  # Set C++ standard settings.
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS ON
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES "86")

# CUDA compile flags for Release only
target_compile_options(${PROJECT_NAME} PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:

  -arch=sm_86
  -gencode=arch=compute_86,code=sm_86
  --default-stream per-thread
  -Xptxas=-v
  --source-in-ptx
  -lineinfo
  >

  $<$<AND:$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>,$<COMPILE_LANGUAGE:CUDA>>:
  --use_fast_math
  >

  $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:
  -gencode=arch=compute_80,code=sm_80
  -gencode=arch=compute_86,code=sm_86
  -gencode=arch=compute_87,code=sm_87
  -gencode=arch=compute_86,code=compute_86
  >

  $<$<COMPILE_LANGUAGE:CXX>:/MP>
)
