project(preimg)

find_package(OpenCV REQUIRED)

add_library(${PROJECT_NAME} STATIC src/preimg.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})