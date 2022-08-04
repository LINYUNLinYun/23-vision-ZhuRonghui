#设置子工程名
project (A2)
#链接库
add_library(${PROJECT_NAME} src/A2.cpp)
#链接包含文件
target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)