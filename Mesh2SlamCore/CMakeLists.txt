
# Set C++ standard and requirements
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Set C/C++ compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -g")


# Collect all .cpp files in Mesh2SlamCore/src
file(GLOB Mesh2SlamCore_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

# Collect all .h files in Mesh2SlamCore/include
file(GLOB Mesh2SlamCore_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)

# Define the Mesh2SlamCore library
add_library(Mesh2SlamCoreLib STATIC ${Mesh2SlamCore_SRC} ${Mesh2SlamCore_HEADERS})

# Include directories for Mesh2SlamCore
include_directories(
        ${CMAKE_SOURCE_DIR}/external/glm
        ${CMAKE_SOURCE_DIR}/Mesh2SlamCore/include
        ${CMAKE_SOURCE_DIR}/external/opencv/jni/include  # Include OpenCV headers
        ${CMAKE_SOURCE_DIR}/external/Eigen3              # Include Eigen3 headers
        ${CMAKE_SOURCE_DIR}/external/g2o/include         # Include g2o headers
)

# OpenCV configuration
include_directories(${CMAKE_SOURCE_DIR}/external/opencv/jni/include/)
set(OpenCV_LIB_DIR ${CMAKE_SOURCE_DIR}/external/opencv/staticlibs/${ANDROID_ABI})
message(STATUS "ANDROID_ABI is set to ${ANDROID_ABI}")
# Static OpenCV libraries
set(OpenCV_LIBS
        ${OpenCV_LIB_DIR}/libopencv_core.a
        ${OpenCV_LIB_DIR}/libopencv_features2d.a
        ${OpenCV_LIB_DIR}/libopencv_imgproc.a
        ${OpenCV_LIB_DIR}/libopencv_highgui.a
        ${OpenCV_LIB_DIR}/libopencv_imgcodecs.a  # Add imgcodecs library
        ${OpenCV_LIB_DIR}/libopencv_videoio.a    # Add videoio library
        # Add more if needed
)

# Define the path to third-party libraries directory
set(THIRD_PARTY_DIR ${CMAKE_SOURCE_DIR}/external/opencv/3rdparty/libs)
# Add all of them
file(GLOB ABI_LIBS ${THIRD_PARTY_DIR}/${ANDROID_ABI}/*.a)

# Eigen3
set(Eigen3_LIB_DIR ${CMAKE_SOURCE_DIR}/external/Eigen3)
include_directories(${Eigen3_LIB_DIR})

# g2o
# Set the path to the g2o library
set(G2O_ROOT ${CMAKE_SOURCE_DIR}/external/g2o)
# Include g2o headers
include_directories(${G2O_ROOT}/include)
# Find all g2o library files
file(GLOB G2O_LIBS ${G2O_ROOT}/lib/*.a)

# glm
include_directories(${CMAKE_SOURCE_DIR}/external/glm)



# Link necessary libraries
target_link_libraries(Mesh2SlamCoreLib
        ${OpenCV_LIBS}  # Link OpenCV libraries
        ${G2O_LIBS}     # Link g2o libraries
        ${ABI_LIBS}     # Link third-party ABI libraries
)
