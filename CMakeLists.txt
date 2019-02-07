project(pix2face_super)
cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_LIBRARY_PATH "$ENV{NVIDIA_LIB_DIR}" CACHE PATH "Direct Cmake to correct OpenGL libs")
set(EGL_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/EGL-Registry/api" CACHE PATH "Path containing EGL header dir")

# libigl wants to install some cmake config files, so give it a local directory to drop them in.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/install CACHE PATH "install prefix" FORCE)
endif()

add_subdirectory(face3d)
