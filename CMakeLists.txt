cmake_minimum_required(VERSION 3.3) # Set minimum CMake version
project(NN)                         # Create project with name "NN"

set(CMAKE_CXX_STANDARD 14)          # Set C++ Standard to 14 to allow threads and mutexs
                                    # Along with 'make_shared' for smart pointers
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Ensure C++14 is used by compiler
set(CMAKE_CXX_EXTENSIONS OFF)       # Ensures C++14 is used and not GNU++14
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/build/debug")
                                    # Set debug directory

set(THREADS_PREFER_PTHREAD_FLAG ON) # Enables pthreading

find_package(Threads REQUIRED)      # Finds threads include packages
include_directories(include)        # Add include directory

file(GLOB sources src/*.cpp)        # Adds all source files to a list

add_executable(NN ${sources})       # Adds the sources list files to the project 

target_link_libraries(NN PRIVATE Threads::Threads)
                                    # Links threads packages and includes to the project