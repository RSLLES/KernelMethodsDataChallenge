cmake_minimum_required(VERSION 3.8)

project(treeEditDistance)

set(CMAKE_CXX_STANDARD 20)

include_directories("D:/Documents_D/Mines/Cours/Master/Kernel/Project/treeEditDistance/lib")

add_executable(treeEditDistance
    lib/hungarian/hungarian.cpp
    main.cpp
)