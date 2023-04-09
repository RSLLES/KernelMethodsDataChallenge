cmake_minimum_required(VERSION 3.0)

project(aptedModule)

set(CMAKE_CXX_STANDARD 11)

include_directories(./)

add_executable(aptedModule
    hungarian.cpp
    hungarian.h
    ted.cpp
    ted.h
    main.cpp)