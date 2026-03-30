cmake_minimum_required(VERSION 3.16)
project(grpc_hotspot_transfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

set(PROTO_FILE "${CMAKE_CURRENT_SOURCE_DIR}/proto/transfer.proto")
set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GENERATED_DIR}")

set(PROTO_SRCS
    "${GENERATED_DIR}/transfer.pb.cc"
    "${GENERATED_DIR}/transfer.grpc.pb.cc")
set(PROTO_HDRS
    "${GENERATED_DIR}/transfer.pb.h"
    "${GENERATED_DIR}/transfer.grpc.pb.h")

add_custom_command(
    OUTPUT ${PROTO_SRCS} ${PROTO_HDRS}
    COMMAND $<TARGET_FILE:protobuf::protoc>
    ARGS
        --proto_path "${CMAKE_CURRENT_SOURCE_DIR}/proto"
        --cpp_out "${GENERATED_DIR}"
        --grpc_out "${GENERATED_DIR}"
        --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
        "${PROTO_FILE}"
    DEPENDS "${PROTO_FILE}"
    COMMENT "Generating gRPC and protobuf sources"
    VERBATIM)

add_library(transfer_proto ${PROTO_SRCS} ${PROTO_HDRS})
target_include_directories(transfer_proto PUBLIC "${GENERATED_DIR}")
target_link_libraries(transfer_proto
    PUBLIC
    protobuf::libprotobuf
    gRPC::grpc++)

add_executable(receiver src/receiver.cpp)
target_link_libraries(receiver PRIVATE transfer_proto)

add_executable(sender src/sender.cpp)
target_link_libraries(sender PRIVATE transfer_proto)
