#pragma once

#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Pre_image{
    public:
    /**
     * @brief bgr->binary  基于inRange阈值过滤得到橙色
     * 
     * @param input,bgr图 
     * @return Mat (二值图)
     */
    Mat findOrange(const Mat input);
    /**
     * @brief 通道相减得到蓝色光
     * 
     * @param input ,bgr图 
     * @return Mat 返回binary
     */
    Mat whiteBlurBlue(const Mat input);	
    /**
     * @brief 通道相减得到红色光
     * 
     * @param input ,bgr图 
     * @return Mat 返回binary
     */
    Mat whiteBlurRed(const Mat input);
    /**
     * @brief bgr->binary  基于inRange阈值过滤得到蓝色
     * 
     * @param input 
     * @return Mat  (二值图)
     */
    static Mat findBlue(const Mat input);
    
    /**
     * @brief 重载函数1：输入Mat和rotaterect画框
     * 
     * @param inputArray 
     * @param inputRotateRect 
     */
    static void line(Mat inputArray,RotatedRect inputRotateRect);
    
    /**
     * @brief 重载函数2：输入四个点画框
     * 
     * @param inputArray 
     * @param ptr 指针，为数组的头地址
     */
    static void line(Mat inputArray,Point2f ptr[]);
    
    /**
     * @brief 输入三个点,判断点b是不是直角
     * 
     * @return true 
     * @return false 
     */
    static bool rightAngle(Point2f a,Point2f b,Point2f c);		
};

