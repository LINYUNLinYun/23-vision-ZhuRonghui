project(armourplate)

find_package(OpenCV REQUIRED)


add_library(${PROJECT_NAME} STATIC src/armourplate.cpp) 

target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} lightbar preimg)