cmake_minimum_required(VERSION 3.10)
project(memefilter)

set(OpenCV_DIR "C:/Users/jessi/OneDrive/Desktop/projects/opencv/build")  # Adjust this path as per your installation

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(memefilter src/Main.cpp)

target_link_libraries(memefilter ${OpenCV_LIBS})
