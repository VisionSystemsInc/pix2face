#project(janus_pvr)
cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_LIBRARY_PATH "$ENV{NVIDIA_LIB_DIR}" CACHE PATH "Direct Cmake to correct OpenGL libs")
set(DLIB_INCLUDE_PATH "${CMAKE_CURRENT_LIST_DIR}/dlib" CACHE PATH "location of dlib source")
set(DLIB_USE_CUDA OFF CACHE BOOL "Don't bother with DLIB cuda build")
set(BUILD_PVR ON CACHE BOOL "Build Janus PVR component")
set(BUILD_PVR_INDEPENDENT ON CACHE BOOL "Build only PVR Janus component")
set(PVR_USE_SYSTEM_LIBS ON CACHE BOOL "Use System Libraries to build PVR")
set(BUILD_JANUS_NET OFF CACHE BOOL "No need to build Janus Deep Network Code")
set(EGL_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/EGL-Registry/api" CACHE PATH "Path containing EGL header dir")

add_subdirectory(janus)
