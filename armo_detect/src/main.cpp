#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "preimg.h"
#include "My_kalman.h"
#include "Lightbar.h"
#include "armourplate.h"
#include "SolvePNP.h"

using namespace cv;
using namespace std;


int main(){
    Mat frame;
    VideoCapture capture;
    My_kalman kf;
    SolvePNP pnp;
    kf.kalman_init();
    capture.open("../2.avi");
    if(!capture.isOpened()){
        cerr<<"wrong open!";
    }
    while(1){
    //frame = imread("../1.png");
    capture>>frame;
    Mat binaryInput = frame.clone();
    Pre_image pImg;
    if(frame.empty()){
        cerr<<"frame wrong!"<<endl;
        break;
    }
    
    Mat binary = pImg.findOrange(binaryInput);
    //Mat binary = pImg.findBlue(binaryInput);
    

    vector<Lightbar> lights;

    lights = Lightbar::screenToContous(binary);
   
    lights = Lightbar::screenToRotate(lights);
    //Mat binary = pImg.whiteBlurBlue(binaryInput);
    // for(int i = 0;lights.size() != 0&&i<lights.size();i++){
    //     Pre_image::line(frame,lights[i].rtRect);

    // }
    vector<Armour> armour1;
    for(int i = 0;i<lights.size();i++){
        for(int j = i + 1;j<lights.size();j++){
            Armour armour(lights[i],lights[j]);
            armour1.push_back(Armour(lights[i],lights[j]));    
        }
    }
    armour1 = Armour::screen(armour1);
    for(int i = 0;i<armour1.size();i++){
        pnp.solution(armour1[i].armourPoints);

        Pre_image::line(frame,armour1[i].armourPoints);
        circle(frame,kf.kalman_predict(armour1[i].armourCenter),20,Scalar(125,0,125),2,8);
    }
    imshow("binary",binary);
    imshow("frame",frame);
    waitKey(0);
    }
    capture.release();
    return 0;
}

