cmake_minimum_required(VERSION 4.0.0)
project(YOLOv11_OBB)

set(ONNXRUNTIME_DIR "/Users/agraw1/Desktop/Fall2025/Intro-to-parallel/project/Code/onnxruntime-osx-arm64-1.23.2")
#set(ONNXRUNTIME_DIR "/opt/homebrew/Cellar/onnxruntime/1.22.2_6")
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

# -------------- Support C++17 for using filesystem  ------------------#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/opencv")
#list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/onnxruntime")

# --- Apple Silicon / macOS fixes ---
# Ensure CMake finds ARM64 libraries (M1/M2/M3)
set(CMAKE_OSX_ARCHITECTURES "arm64")

find_package(OpenCV REQUIRED)
# Debug message
message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(onnxruntime REQUIRED)
message(STATUS "Found onnxruntime version: ${onnxruntime_VERSION}")
message(STATUS "onnxruntime include path: ${onnxruntime_INCLUDE_DIRS}")
message(STATUS "onnxruntime libraries: ${onnxruntime_LIBS}")
include_directories(${onnxruntime_INCLUDE_DIRS})

include_directories("include/")

# Add executable for image inference
add_executable(image_inference
               src/image_inference.cpp)

# Add executable for camera inference
add_executable(camera_inference
               src/camera_inference.cpp)

# Add executable for video inference
add_executable(video_inference
               src/video_inference.cpp)

# Set include directories for all executables
target_include_directories(image_inference PRIVATE "${ONNXRUNTIME_DIR}/include")
target_include_directories(camera_inference PRIVATE "${ONNXRUNTIME_DIR}/include")
target_include_directories(video_inference PRIVATE "${ONNXRUNTIME_DIR}/include")

# Set compile features for all executables
target_compile_features(image_inference PRIVATE cxx_std_17)
target_compile_features(camera_inference PRIVATE cxx_std_17)
target_compile_features(video_inference PRIVATE cxx_std_17)

# Link libraries for all executables
target_link_libraries(image_inference ${OpenCV_LIBS})
target_link_libraries(camera_inference ${OpenCV_LIBS})
target_link_libraries(video_inference ${OpenCV_LIBS})

#if(UNIX)
#    message(STATUS "We are building on Linux!")
#    # Specific Linux build commands or flags
#    target_link_libraries(image_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
#    target_link_libraries(camera_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
#    target_link_libraries(video_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
#endif(UNIX)

if(APPLE)
    message(STATUS "We are building on macOS!")
    # Specific macOS build commands or flags
    target_link_libraries(image_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
    target_link_libraries(camera_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
    target_link_libraries(video_inference "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
endif(APPLE)

if(WIN32)
    message(STATUS "We are building on Windows!")
    # Specific Windows build commands or flags
    target_link_libraries(image_inference "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
    target_link_libraries(camera_inference "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
    target_link_libraries(video_inference "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
endif(WIN32)
