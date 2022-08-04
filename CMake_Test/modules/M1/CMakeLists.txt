#设置子工程名
project (M1)
#链接库
add_library(${PROJECT_NAME} src/M1.cpp)
#链接包含文件
target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    A1
)
#链接相关库
target_link_libraries(${PROJECT_NAME} A1)
