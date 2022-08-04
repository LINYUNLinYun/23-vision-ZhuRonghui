#pragma once

#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "Lightbar.h"
#include "preimg.h"


using namespace std;
using namespace cv;

class Armour{
public:
    Armour(Lightbar lightbar1,Lightbar lightbar2);
    //灯条的四个角点
    Point2f armourPoints[4];
    //灯条的中心点
    Point2f armourCenter;
    //装甲板的宽
    float upWidth;
    //装甲板的宽
    float downWidth;
    //装甲板的高
    float leftHeight;
    //装甲板的高
    float rightHeight;
    /**
     * @brief 过滤函数
     * 
     * @param inputArmour 
     * @return vector<Armour> 
     */
    static vector<Armour> screen(vector<Armour> inputArmour);
};