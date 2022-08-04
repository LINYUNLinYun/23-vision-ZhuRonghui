project(Armo)

cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)  

find_package(OpenCV REQUIRED)

add_subdirectory(image_preprocessing)
add_subdirectory(my_kalman)
add_subdirectory(lightbar)
add_subdirectory(armourplate)
add_subdirectory(serialport)
add_subdirectory(solvepnp)

add_executable(${PROJECT_NAME} src/main.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} preimg kalman lightbar armourplate solvepnp)

