cmake_minimum_required(VERSION 3.14)
project(CPython)

file(GLOB sources
        Modules/*.c
        Objects/*.c
        Python/*.c
        Programs/*.c
        )

configure_file("${PROJECT_SOURCE_DIR}/pyconfig.h.in" "${PROJECT_SOURCE_DIR}/pyconfig.h")
include_directories(.)
include_directories(Include)
include_directories(Include/internal)

add_executable(${PROJECT_NAME}
        ${sources}
        )
