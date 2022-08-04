#设置子工程名
project (M2)
find_package(OpenCV REQUIRED)
#链接源文件
add_library(${PROJECT_NAME} src/M2.cpp)
#链接包含文件
target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    A1 
    A2 
    ${kalman_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIR}
)
#链接相关库
target_link_libraries(${PROJECT_NAME} A1 A2 ${OpenCV_LIBS})