#设置子工程名
project (A1)
#链接源文件，将指定的源文件生成链接文件
aux_source_directory(src A1_DIRS)
#生成动态库
add_library(${PROJECT_NAME} SHARED ${A1_DIRS})
#添加包含文件
target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)