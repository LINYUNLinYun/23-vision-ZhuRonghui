#include "Lightbar.h"

Lightbar::Lightbar(vector<Point> &singleContours){
    rtRect = minAreaRect(singleContours);
    lightCenter = rtRect.center;
    Point2f anglePoints[4];
    rtRect.points(anglePoints);
    //重构函数
    if(rtRect.size.height<rtRect.size.width){
        rtRect = RotatedRect(anglePoints[1],anglePoints[2],anglePoints[3]);
    }
    //刷新角点
    rtRect.points(anglePoints);
    upCenter.x = 0.5*(anglePoints[1].x+anglePoints[2].x);
    upCenter.y = 0.5*(anglePoints[1].y+anglePoints[2].y);
    downCenter.x = 0.5*(anglePoints[0].x+anglePoints[3].x);
    downCenter.y = 0.5*(anglePoints[0].y+anglePoints[3].y);
}

vector<Lightbar> Lightbar::screenToRotate(vector<Lightbar> inputLights){
    //临时变量
    vector<Lightbar> tempLights;
    for (int i = 0; i < inputLights.size(); i++)
    {
        float width = inputLights[i].rtRect.size.width;
        float height = inputLights[i].rtRect.size.height;
        float hw_rate = height > width ? height / width : width / height;
        if(hw_rate<2.8){
            continue;
        }
        tempLights.push_back(inputLights[i]);
    }
    //交换临时和输入，临时清零
    swap(tempLights,inputLights);
    tempLights.clear();
    return inputLights;
}

vector<Lightbar> Lightbar::screenToContous(Mat inputBinary){
    //从binary中提取出来的轮廓信息
    vector<vector<Point>>contours;
    //凸包的点集信息
    vector<Point> hullPoints;
    vector<Vec4i>hierachy;
    vector<Lightbar> templights;
    //使轮廓信息平滑
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	erode(inputBinary, inputBinary, element, Point(-1, -1), 1);
	dilate(inputBinary, inputBinary, element, Point(-1, -1), 1);
    //找出轮廓
    findContours(inputBinary, contours, hierachy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    //粗过滤
    for(int i = 0;i<contours.size();i++){
    //1.相对面积过滤
        RotatedRect minArea = minAreaRect(contours[i]);
        float area = contourArea(contours[i]);
        float areaRate = minArea.size.area()/area;
        if(areaRate>1.5||areaRate<1){
            continue;
        }
        //用凸缺陷衡量轮廓的规整度
        convexHull(contours[i], hullPoints, false);
        float solidity = contourArea(hullPoints) / area;
        if(solidity<0.8){
            continue;
        }
        templights.push_back(Lightbar(contours[i]));
    }
    return templights; 
}

