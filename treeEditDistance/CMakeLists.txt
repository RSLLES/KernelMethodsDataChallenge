cmake_minimum_required(VERSION 3.0)

project(aptedModule)

set(CMAKE_CXX_STANDARD 11)

include_directories("D:/Documents_D/Mines/Cours/Master/Kernel/Project/treeEditDistance/lib")

add_executable(aptedModule
    lib/hungarian/hungarian.cpp
    main.cpp
)