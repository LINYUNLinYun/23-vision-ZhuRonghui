#设置子工程名
project (math)
#找包
find_package(OpenCV REQUIRED)
#添加源文件
add_library(${PROJECT_NAME} SHARED src/Math.cpp)
#添加包含文件
target_include_directories(${PROJECT_NAME} PUBLIC 
${PROJECT_SOURCE_DIR}/include
${OpenCV_INCLUDE_DIR})
#链接库
target_link_libraries(${PROJECT_NAME}  ${OpenCV_LIBS})

