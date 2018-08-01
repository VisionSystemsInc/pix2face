project(pix2face_super)
cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_LIBRARY_PATH "$ENV{NVIDIA_LIB_DIR}" CACHE PATH "Direct Cmake to correct OpenGL libs")
set(EGL_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/EGL-Registry/api" CACHE PATH "Path containing EGL header dir")


add_subdirectory(face3d)
