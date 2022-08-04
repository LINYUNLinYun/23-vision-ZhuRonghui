#pragma once

#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Lightbar
{
private:
    
public:
    //灯条的最小旋转矩形  
    RotatedRect rtRect;
    //灯条的中心点
    Point2f lightCenter;
    //灯条上顶点
    Point2f upCenter;
    //灯条下顶点
    Point2f downCenter;
    /**
     * @brief Construct a new Lightbar object
     * 
     * @param singleContours 
     */
    Lightbar(vector<Point> &singleContours);

    /**
     * @brief 静态函数，粗过滤，返回疑似灯条的轮廓
     * 
     * @param inputBinary
     * @return vector<Lightbar> 
     */
    static vector<Lightbar> screenToContous(Mat inputBinary);

    /**
     * @brief 静态函数，灯条过滤函数
     * 
     * @return vector<Lightbar> 
     */
    static vector<Lightbar> screenToRotate(vector<Lightbar> inputLights);
    
};

