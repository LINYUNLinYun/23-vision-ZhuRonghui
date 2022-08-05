#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "preimg.h"
#include "My_kalman.h"
#include "Lightbar.h"
#include "armourplate.h"
#include "SolvePNP.h"
#include "SerialPort.h"

using namespace cv;
using namespace std;

//全局变量
bool isLockTaget = true;

/**
 * @brief 锁定某个装甲板函数
 * 
 * @param inputarmour 
 * @param target 
 * @return vector<Armour> 
 */
vector<Armour> selectTarget(vector<Armour> inputarmour,Point2f target = Point2f(320,240));

/**
 * @brief 鼠标回调函数
 * 
 * @param event 
 * @param x 
 * @param y 
 * @param flags 
 * @param param 
 */
void onMouse(int event, int x, int y, int flags, void* param){
    if(event == EVENT_RBUTTONDOWN){
        isLockTaget = !isLockTaget;
    }
}

int main(){
    char key = 0;
    Mat frame;
    VideoCapture capture;
    My_kalman kf;
    SolvePNP pnp;
    SerialPort UART;
    //发送到stm32的数据
    vector<unsigned char> temptx;
    kf.kalman_init();
    UART.SerialPort_init(UART);
    capture.open("../2.avi");
    if(!capture.isOpened()){
        cerr<<"wrong open!";
    }
    while(1){
        if(UART.Open()==false){
            cerr<<"UART open error"<<endl;
        }
	    else cerr<<"OPEN OK!\n";
            if(UART.LoadConfig()==false) cerr<<"Set error!\n";
	    else cout<<"Set OK!\n";
        //capture>>frame;
        frame = imread("../1.png");
        Mat binaryInput = frame.clone();
        Pre_image pImg;
        if(frame.empty()){
            cerr<<"frame wrong!"<<endl;
            break;
        }

        //Mat binary = pImg.findOrange(binaryInput);
        Mat binary = pImg.findBlue(binaryInput);


        vector<Lightbar> lights;

        lights = Lightbar::screenToContous(binary);
    
        lights = Lightbar::screenToRotate(lights);

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
        }
        if(armour1.size()){
        armour1 = selectTarget(armour1,Point2f(frame.cols,frame.rows));
        circle(frame,kf.kalman_predict(armour1[0].armourCenter),20,Scalar(125,0,125),2,8);
        if(isLockTaget){
            cout<<"目标已锁定！！！"<<endl;
            temptx = UART.SendMessageChange(pnp.yaw,pnp.pitch);
            //输出流重载,发送yaw和pitch给云台
            UART<<temptx;
            
        }
        }
        imshow("binary",binary);
        namedWindow("frame",WINDOW_AUTOSIZE);
        imshow("frame",frame);
        setMouseCallback("frame",onMouse);
        key = waitKey(20);
        if(key == 27){
            break;
        }
    }
    UART.Close();
    capture.release();
    return 0;
}

vector<Armour> selectTarget(vector<Armour> inputarmour,Point2f target){
    //距离容器
    float dis = 1000000;
    int index = 0;
    for(int i = 0;i<inputarmour.size();i++){
    float distance = sqrtf(powf((inputarmour[i].armourCenter.x - target.x),2) + powf((inputarmour[i].armourCenter.y - target.y),2));
       if(distance < dis){
        dis = distance;
        index = i;
       }
    }
    vector<Armour> temp;
    temp.push_back(inputarmour[index]);
    return temp;
}
