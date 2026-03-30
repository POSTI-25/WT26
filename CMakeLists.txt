cmake_minimum_required(VERSION 3.16)
project(zmq_hotspot_transfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBZMQ REQUIRED libzmq)

add_executable(receiver src/receiver.cpp)
target_include_directories(receiver PRIVATE ${LIBZMQ_INCLUDE_DIRS})
target_link_libraries(receiver PRIVATE ${LIBZMQ_LIBRARIES})
target_compile_options(receiver PRIVATE ${LIBZMQ_CFLAGS_OTHER})

add_executable(sender src/sender.cpp)
target_include_directories(sender PRIVATE ${LIBZMQ_INCLUDE_DIRS})
target_link_libraries(sender PRIVATE ${LIBZMQ_LIBRARIES})
target_compile_options(sender PRIVATE ${LIBZMQ_CFLAGS_OTHER})
