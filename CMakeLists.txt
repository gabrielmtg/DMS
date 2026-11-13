cmake_minimum_required(VERSION 3.15)
project(DriverMonitoringPipeline CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Encontra o OpenCV local do seu sistema
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# Diretório de headers
include_directories(${PROJECT_SOURCE_DIR}/include)

# Biblioteca apenas com a lógica de crop
add_library(core_pipeline src/preprocessor.cpp)
target_link_libraries(core_pipeline PRIVATE ${OpenCV_LIBS})

# Executável de teste
add_executable(test_crop tests/test_crop.cpp)
target_link_libraries(test_crop PRIVATE core_pipeline ${OpenCV_LIBS})