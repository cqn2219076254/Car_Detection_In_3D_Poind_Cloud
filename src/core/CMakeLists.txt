cmake_minimum_required(VERSION 3.1)
project(hci_project)

set(CMAKE_CXX_STANDARD 14)
set(OpenCV_DIR "/home/ma-user/work/opencv-4.5.1/build")
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
find_package(OpenCV REQUIRED)
include_directories(
        ${PROJECT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
)

add_executable(main tester.cpp)
target_link_libraries(main ${OpenCV_LIBS})
