# Copyright 2023, The Khronos Group Inc.
#
# SPDX-License-Identifier: Apache-2.0

# XR_DOCS_TAG_BEGIN_SetProjectName3
cmake_minimum_required(VERSION 3.22.1)
set(PROJECT_NAME Mesh2Slam)
project("${PROJECT_NAME}")
# XR_DOCS_TAG_END_SetProjectName3


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Specify the STL to use (c++_shared or c++_static)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DANDROID_STL=c++_shared")


# XR_DOCS_TAG_BEGIN_CMakeModulePath
# Additional Directories for find_package() to search within.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
# XR_DOCS_TAG_END_CMakeModulePath
include("../cmake/graphics_api_select.cmake")

# XR_DOCS_TAG_BEGIN_FetchContent
# For FetchContent_Declare() and FetchContent_MakeAvailable()
include(FetchContent)

# openxr_loader - From github.com/KhronosGroup
set(BUILD_TESTS
    OFF
    CACHE INTERNAL "Build tests"
)
set(BUILD_API_LAYERS
    ON
    CACHE INTERNAL "Use OpenXR layers"
)
FetchContent_Declare(
    OpenXR
    URL_HASH MD5=924a94a2da0b5ef8e82154c623d88644
    URL https://github.com/KhronosGroup/OpenXR-SDK-Source/archive/refs/tags/release-1.0.34.zip
        SOURCE_DIR
        openxr
)
FetchContent_MakeAvailable(OpenXR)
# XR_DOCS_TAG_END_FetchContent

# Files
set(SOURCES
    main.cpp
    ../Common/GraphicsAPI.cpp
    ../Common/GraphicsAPI_D3D11.cpp
    ../Common/GraphicsAPI_D3D12.cpp
    ../Common/GraphicsAPI_OpenGL.cpp
    ../Common/GraphicsAPI_OpenGL_ES.cpp
    ../Common/GraphicsAPI_Vulkan.cpp
    ../Common/OpenXRDebugUtils.cpp)
set(HEADERS
    ../Common/DebugOutput.h
    ../Common/GraphicsAPI.h
    ../Common/GraphicsAPI_D3D11.h
    ../Common/GraphicsAPI_D3D12.h
    ../Common/GraphicsAPI_OpenGL.h
    ../Common/GraphicsAPI_OpenGL_ES.h
    ../Common/GraphicsAPI_Vulkan.h
    ../Common/HelperFunctions.h
    ../Common/OpenXRDebugUtils.h
    ../Common/OpenXRHelper.h)

# XR_DOCS_TAG_BEGIN_HLSLShaders
set(HLSL_SHADERS
    "../Shaders/VertexShader.hlsl"
    "../Shaders/PixelShader.hlsl")
# XR_DOCS_TAG_END_HLSLShaders
# XR_DOCS_TAG_BEGIN_GLSLShaders
set(GLSL_SHADERS
    "../Shaders/VertexShader.glsl"
    "../Shaders/PixelShader.glsl")
# XR_DOCS_TAG_END_GLSLShaders
# XR_DOCS_TAG_BEGIN_GLESShaders
set(ES_GLSL_SHADERS
    "../Shaders/VertexShader_GLES.glsl"
    "../Shaders/PixelShader_GLES.glsl")
# XR_DOCS_TAG_END_GLESShaders


    # XR_DOCS_TAG_BEGIN_Android

    #Mesh2SlamCore
    include_directories( ${CMAKE_SOURCE_DIR}/Mesh2SlamCore/include)
    # Collect all .cpp files in Mesh2SlamCore/src
    file(GLOB Mesh2SlamCore_SRC ${CMAKE_SOURCE_DIR}/Mesh2SlamCore/src/*.cpp)

    # Copy shader source files to the build directory during build
    file(GLOB SHADER_FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.vert"
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.frag")
    file(COPY ${SHADER_FILES} DESTINATION "${CMAKE_BINARY_DIR}/shaders")


    add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS} ${Mesh2SlamCore_SRC})
    target_include_directories(${PROJECT_NAME} PRIVATE
        # In this repo
        ../Common/
        # From OpenXR repo
        "${openxr_SOURCE_DIR}/src/common"
        "${openxr_SOURCE_DIR}/external/include"
    )

    #OpenCV STUFF
    include_directories(${CMAKE_SOURCE_DIR}/external/opencv/jni/include/)
    set(OpenCV_LIB_DIR ${CMAKE_SOURCE_DIR}/external/opencv/staticlibs/${ANDROID_ABI})
    message(STATUS "ANDROID_ABI is set to ${ANDROID_ABI}")
    #static opencv libraries
    set(OpenCV_LIBS
    ${OpenCV_LIB_DIR}/libopencv_core.a
    ${OpenCV_LIB_DIR}/libopencv_features2d.a
    ${OpenCV_LIB_DIR}/libopencv_imgproc.a
    ${OpenCV_LIB_DIR}/libopencv_highgui.a
    ${OpenCV_LIB_DIR}/libopencv_imgcodecs.a  # Add imgcodecs library
    ${OpenCV_LIB_DIR}/libopencv_videoio.a    # Add videoio library
    #add more if needed
    )

    # Define the path to third-party libraries directory
    set(THIRD_PARTY_DIR ${CMAKE_SOURCE_DIR}/external/opencv/3rdparty/libs)
    # add all of them
    file(GLOB ABI_LIBS ${THIRD_PARTY_DIR}/${ANDROID_ABI}/*.a)

    #Eigen3
    set(Eigen3_LIB_DIR ${CMAKE_SOURCE_DIR}/external/Eigen3)
    include_directories(${Eigen3_LIB_DIR})

    #g2o
    # Set the path to the g2o library
    set(G2O_ROOT ${CMAKE_SOURCE_DIR}/external/g2o)

    # Include g2o headers
    include_directories(${G2O_ROOT}/include)

    # Find all g2o library files
    file(GLOB G2O_LIBS ${G2O_ROOT}/lib/*.so)

    #glm
    include_directories( ${CMAKE_SOURCE_DIR}/external/glm)

    add_subdirectory(Mesh2SlamCore)



    # export ANativeActivity_onCreate for java to call.
    set_property(
        TARGET ${PROJECT_NAME}
        APPEND_STRING
        PROPERTY LINK_FLAGS " -u ANativeActivity_onCreate"
    )

    # native_app_glue
    include(AndroidNdkModules)
    android_ndk_import_module_native_app_glue()

    find_package(ZLIB REQUIRED)
    if (ZLIB_FOUND)
        include_directories(${ZLIB_INCLUDE_DIRS})
        list(APPEND OpenCV_LIBS ${ZLIB_LIBRARIES})
    else()
        message(FATAL_ERROR "zlib not found")
    endif()



    target_link_libraries(${PROJECT_NAME}
            android
            native_app_glue
            openxr_loader
            ${OpenCV_LIBS}     # Link OpenCV libraries
            ${ABI_LIBS}        # Link ABI third party libraries
            ${G2O_LIBS}        # Link ABI third party libraries
            ${ZLIB_LIBRARIES}
    )
    target_compile_options(${PROJECT_NAME} PRIVATE -Wno-cast-calling-convention)
    # XR_DOCS_TAG_END_Android
    AddGraphicsAPIDefine(${PROJECT_NAME})


    # XR_DOCS_TAG_BEGIN_Gfxwrapper
    include(../cmake/gfxwrapper.cmake)
    if(TARGET openxr-gfxwrapper)
        target_include_directories(${PROJECT_NAME} PUBLIC ${openxr_SOURCE_DIR}/src/common)
        target_link_libraries(${PROJECT_NAME} openxr-gfxwrapper)
        target_compile_definitions(${PROJECT_NAME} PUBLIC XR_TUTORIAL_USE_OPENGL_ES)
    endif()
    # XR_DOCS_TAG_END_Gfxwrapper


    #shaders
    set(SHADERS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/Mesh2SlamCore/shaders")
    set(SHADERS_DEST "${CMAKE_CURRENT_SOURCE_DIR}/app/src/main/assets/shaders")
    file(MAKE_DIRECTORY ${SHADERS_DEST})
    file(GLOB SHADERS_FILES "${SHADERS_SRC}/*")
    file(COPY ${SHADERS_FILES} DESTINATION "${SHADERS_DEST}")

    #config file
    set(CONFIGFILE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/Mesh2SlamCore/include/slamConfigFile.yaml")
    set(CONFIGFILE_DEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/app/src/main/assets/configFile")
    file(MAKE_DIRECTORY ${CONFIGFILE_DEST_DIR})
    file(COPY ${CONFIGFILE_SRC} DESTINATION ${CONFIGFILE_DEST_DIR})

    #dataframes
    set(DATAFRAMES_SRC "${CMAKE_CURRENT_SOURCE_DIR}/dataFrames")
    set(DATAFRAMES_DEST "${CMAKE_CURRENT_SOURCE_DIR}/app/src/main/assets/dataFrames")
    file(MAKE_DIRECTORY ${DATAFRAMES_DEST})
    file(GLOB DATAFRAME_FILES "${DATAFRAMES_SRC}/*")
    file(COPY ${DATAFRAME_FILES} DESTINATION "${DATAFRAMES_DEST}")

    #meshes
set(MESHES_SRC "${CMAKE_CURRENT_SOURCE_DIR}/meshes")
set(MESHES_DEST "${CMAKE_CURRENT_SOURCE_DIR}/app/src/main/assets/meshes")
file(MAKE_DIRECTORY ${MESHES_DEST})
file(GLOB MESHES_FILES "${MESHES_SRC}/*")
file(COPY ${MESHES_FILES} DESTINATION "${MESHES_DEST}")
