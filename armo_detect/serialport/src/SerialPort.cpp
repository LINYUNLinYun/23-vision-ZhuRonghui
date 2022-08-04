/***********************************************************************************************************************
* 类名：SerialPort
* 功能描述：用于linux开发板（USB）到stem32（GPIO）的串口通信（需要USB转TTL模块）
* 作者：朱荣辉（该类在github原作者"spray0"的程序上修改而成)
* 修改日期：暂无
* 修改原因：暂无
************************************************************************************************************************/

#include "SerialPort.h"

/*********************************************************************************************
* 函数名：串口初始化函数  SerialPort_init
* 函数功能描述：对串口所需要的波特率、数据位、串口分配文件等进行设置
* 函数参数：传入你需要设置类的实例
* 函数返回值：bool
* 作者：朱荣辉
* 修改日期：暂无
* 修改原因：暂无
* 旁注：当前设置仅基于作者的使用环境，如果你要打开其他的串口或设置不同的参数，可以根据需要修改Config
* 结构中的参数（作者使用ttyUSB0,波特率115200，数据位8，停止位0为例）
*********************************************************************************************/

bool SerialPort::SerialPort_init(SerialPort &UART){
	UART.Config.BaudRate=SerialPort::BR115200;		//波特率115200
    UART.Config.DataBits=SerialPort::DataBits8;		//数据位8
    UART.Config.StopBits=SerialPort::StopBits1;		//停止位0
    UART.Config.Parity=SerialPort::ParityNone;		//校验位0
    UART.Config.DevicePath=(char*)&"/dev/ttyUSB0";	//设备名称
	return true;
}


/*********************************************************************************************
* 函数名：串口发送信息函数  SendMessage
* 函数功能描述：为了将装甲板的位置传送到下位机stm32，因此通过调用串口来发送目标的偏航角和俯仰角。
* 函数参数：angleX:偏航角，angleY:俯仰角
* 函数返回值：bool
* 作者：朱荣辉
* 修改日期：2022.6.14
* 修改原因：进一步完善功能
* 旁注：由于舵机的精度不够，只能进行至少1度的调整，因此这里的行参就直接截断成整型了，若有需要可更改
*********************************************************************************************/
vector<unsigned char> SerialPort::SendMessage(int angleX,int angleY){
    stringstream x1;								//使用sstream和string来修剪数据
    stringstream y1;
    string temp;
    string message;
	vector<unsigned char> res;						//暂存数据的容器
    message.push_back('#');
    if (angleX > 0) { message.push_back('+'); }		//正
    else {
            message.push_back('-');					//负
    }
    x1  << setw(2) << setfill('0')<< abs(angleX);	//控制数据的位数为2，与电控统一
    x1 >> temp;
    message += temp;
    if (angleY > 0) { message.push_back('+'); }
    else {
            message.push_back('-');
    }
    y1  << setw(2) << setfill('0')<< abs(angleY);
    y1 >> temp;
    message += temp;
    message.push_back('%');
	for(int i = 0;i<message.size();i++){
			res.push_back(message[i]);
	}
    return res;			//返回容器
}

//接收回调函数
void __attribute__((weak)) RxData_CallBack(std::vector<unsigned char> &data, int fd) {
	for (auto c : data)
		printf("%c", c);
}
//监听线程　读取的数据存放在容器
void* Listen(void *arg) {
	int get;
	int fd = *((int*) arg);
	std::vector<unsigned char> RX_buf(128);
	while (1) {
		get = read(fd, &RX_buf[0], 128);
		if (get > 0) {
			std::vector<unsigned char> RX_data;
			for (int c = 0; c < get; ++c)
				RX_data.push_back(RX_buf[c]);
			RxData_CallBack(RX_data, fd);
			RX_data.clear();
		}
	}
	return NULL;
}

//发送单个字节
bool SerialPort::Send(unsigned char byte) {
	return (write(this->fd, &byte, 1) == -1) ? false : true;
}
bool operator <<(SerialPort port, unsigned char byte) {
	return (write(port.fd, &byte, 1) == -1) ? false : true;
}
//发送多个字节
bool SerialPort::Send(std::vector<unsigned char> data) {
	return (write(this->fd, &data[0], data.size()) == -1) ? false : true;
}
bool operator <<(SerialPort port, std::vector<unsigned char> data) {
	return (write(port.fd, &data[0], data.size()) == -1) ? false : true;
}
//发送多个字节
bool SerialPort::Send(char *data, unsigned int len) {
	return (write(this->fd, data, len) == -1) ? false : true;
}
bool operator <<(SerialPort port, char const *data) {
	return (write(port.fd, data, strlen(data)) == -1) ? false : true;
}

//打开串口
bool SerialPort::Open() {
	//打开串口
	this->fd = open(this->Config.DevicePath, O_RDWR | O_NOCTTY | O_NONBLOCK);
	if (this->fd == -1)
		return false;
	if (fcntl(this->fd, F_SETFL, 0) < 0)
		return false;
	if (isatty(STDIN_FILENO) == 0)
		return false;

	//清空缓存
	tcflush(fd, TCIOFLUSH);
	fcntl(fd, F_SETFL, 0);

	//开启监听线程
	pthread_create(&this->listen_thread, NULL, Listen, &this->fd);
	pthread_detach(this->listen_thread);

	return true;
}

//关闭串口
void SerialPort::Close() {
	//关闭串口入口
	close(this->fd);
	//关闭监听线程
	pthread_cancel(this->listen_thread);
}

//配置串口
bool SerialPort::LoadConfig() {
	//设置参数
	struct termios newtio, oldtio;
	if (tcgetattr(fd, &oldtio) != 0)
		return false;
	bzero(&newtio, sizeof(newtio));
	newtio.c_cflag |= CLOCAL | CREAD;
	newtio.c_cflag &= ~CSIZE;

	//数据位
	switch (this->Config.DataBits) {
	case this->DataBits7:
		newtio.c_cflag |= CS7;
		break;
	case this->DataBits8:
		newtio.c_cflag |= CS8;
		break;
	}

	//奇偶校验位
	switch (this->Config.Parity) {
	case this->ParityEven:
		newtio.c_iflag |= (INPCK | ISTRIP);
		newtio.c_cflag |= PARENB;
		newtio.c_cflag &= ~PARODD;
		break;
	case this->ParityNone:
		newtio.c_cflag &= ~PARENB;
		break;
	case this->ParityOdd:
		newtio.c_cflag |= PARENB;
		newtio.c_cflag |= PARODD;
		newtio.c_iflag |= (INPCK | ISTRIP);
		break;
	}

	//波特率
	cfsetispeed(&newtio, this->Config.BaudRate);
	cfsetospeed(&newtio, this->Config.BaudRate);

	//停止位
	switch (this->Config.StopBits) {
	case this->StopBits1:
		newtio.c_cflag &= ~CSTOPB;
		break;
	case this->StopBits2:
		newtio.c_cflag |= CSTOPB;
		break;
	}

	newtio.c_cc[VTIME] = 0;
	newtio.c_cc[VMIN] = 0;
	tcflush(fd, TCIOFLUSH);
	fcntl(fd, F_SETFL, 0);
	if ((tcsetattr(fd, TCSANOW, &newtio)) != 0)
		return false;

	return true;
}