cmake_minimum_required(VERSION 3.20)
project(llm_engine_onnx_grpc_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

# Point this at an ONNX Runtime installation when enabling real model execution.
# find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h)
# find_library(ONNXRUNTIME_LIBRARY onnxruntime)

add_executable(onnx_grpc_server
    onnx_grpc_server.cc
    onnx_runtime_core.cc
    generated/inference.pb.cc
    generated/inference.grpc.pb.cc
)

target_include_directories(onnx_grpc_server PRIVATE generated)

target_link_libraries(onnx_grpc_server
    PRIVATE
    gRPC::grpc++
    protobuf::libprotobuf
    # ${ONNXRUNTIME_LIBRARY}
)

# Example generation command from the repository root:
# python -m grpc_tools.protoc \
#   -I proto \
#   --cpp_out=cpp_backend/generated \
#   --grpc_out=cpp_backend/generated \
#   --plugin=protoc-gen-grpc="$(which grpc_cpp_plugin)" \
#   proto/inference.proto
