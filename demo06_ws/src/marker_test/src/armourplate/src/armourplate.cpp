#include "armourplate.h"

Armour::Armour(Lightbar lightbar1,Lightbar lightbar2){
    //保证左1右2
    if(lightbar1.rtRect.center.x > lightbar2.rtRect.center.x){
        swap(lightbar1,lightbar2);
    }
    armourPoints[0] = lightbar1.downCenter;
    armourPoints[1] = lightbar1.upCenter;
    armourPoints[2] = lightbar2.upCenter;
    armourPoints[3] = lightbar2.downCenter;

    armourCenter = Point2f((lightbar1.rtRect.center.x + lightbar2.rtRect.center.x)*0.5,
                            (lightbar1.rtRect.center.y + lightbar2.rtRect.center.y)*0.5);;
    //左右高度
    leftHeight = sqrt(pow(abs(armourPoints[1].x-armourPoints[0].x),2)+pow(abs(armourPoints[1].y-armourPoints[0].y),2));
    rightHeight = sqrt(pow(abs(armourPoints[3].x-armourPoints[2].x),2)+pow(abs(armourPoints[3].y-armourPoints[2].y),2));
    //装甲板的宽
    upWidth = sqrt(pow(abs(armourPoints[1].x-armourPoints[2].x),2)+pow(abs(armourPoints[1].y-armourPoints[2].y),2));
    downWidth = sqrt(pow(abs(armourPoints[0].x-armourPoints[3].x),2)+pow(abs(armourPoints[0].y-armourPoints[3].y),2));
}

vector<Armour> Armour::screen(vector<Armour> inputArmour){
    vector<Armour> tempArmour;
    float heightRate;
    float widthRate;
        cout<<"210"<<inputArmour.size()<<endl;
    for(int i = 0;i<inputArmour.size();i++){
        //装甲的角点应该近似是90度（80-100）
        if(!Pre_image::rightAngle(inputArmour[i].armourPoints[0],inputArmour[i].armourPoints[1],inputArmour[i].armourPoints[2])){
            continue;
        }
        heightRate = inputArmour[i].leftHeight>inputArmour[i].rightHeight?
                     inputArmour[i].leftHeight/inputArmour[i].rightHeight:
                     inputArmour[i].rightHeight/inputArmour[i].leftHeight;
        //装甲的两个灯条应该近似等长，则长度比超过1.5，匹配失败
        if(heightRate>1.5){
            continue;
        }
        //装甲的两个灯条应该差不多平行，则上下底边距比超过1.5，匹配失败
        widthRate = inputArmour[i].upWidth>inputArmour[i].downWidth?
                    inputArmour[i].upWidth/inputArmour[i].downWidth:
                    inputArmour[i].downWidth>inputArmour[i].upWidth;
        if(widthRate>1.5){
            continue;
        }
        //装甲板的长宽比应该在1.5-3.2
        double ratio = (inputArmour[i].upWidth+inputArmour[i].downWidth)/(inputArmour[i].leftHeight+inputArmour[i].rightHeight);
        //cout<<"  width  "<<inputArmour[i].upWidth+inputArmour[i].downWidth<<endl;
        //cout<<"  height  "<<inputArmour[i].leftHeight+inputArmour[i].rightHeight<<endl;
        if(ratio>3.2||ratio<1.5){
            continue;
        }
        //cout<<"4444444"<<endl;

        tempArmour.push_back(inputArmour[i]);
    }
    return tempArmour;
}