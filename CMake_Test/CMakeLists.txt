# 在此处进行 CMake 设置（最小版本号，项目名）
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)  #C++17
project(Test)
#find_package(OpenCV REQUIRED)
#添加子目录
####### 以下禁止修改 #######
add_subdirectory(common)   #
add_subdirectory(modules)  #
####### 以上禁止修改 #######

# 在此处生成可执行程序，添加源代码
add_executable(test main.cpp)

# 在此处链接库
target_link_libraries(test 
    M1
    M2
    math
)