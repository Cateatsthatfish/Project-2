#include<opencv2/opencv.hpp>
#include<iostream>
#include"datas.h" // 数据
#include<cmath> //指数
#include<chrono> //计时
#include<fstream> // write data(用不到)
// o3加速
//#pragma GCC optimize(3, "Ofast", "inline") 
using namespace cv;
using namespace std;



class Layer {
private:
    int number; // 第几个（个数）
    int channel;
    int height;
    int weight;
    float* data; // 因为并不知道有多少个所以只能用动态数组
public:
    /* ---------------------- 基本成员函数 -------------------------*/
        // 默认构造 ： 什么都没有
    Layer() :number(0), channel(0), height(0), weight(0), data(nullptr) {};
    // 初始化 ： 设定数量、通道、宽和高
    Layer(int n, int c, int h, int w, const float* d) :number(n), channel(c), height(h), weight(w) {
        //cout << "initial!" << endl; 这个初始化用了12次
        data = new float[n * c * h * w]();
        for (int i = 0; i < n * c * h * w; i++) {
            data[i] = d[i];
        }
    }
    //拷贝构造
    Layer(const Layer& L) :number(L.number), channel(L.channel), height(L.height), weight(L.weight) {
        //cout << "it is called!" << endl;
        int t = number * channel * height * weight;
        data = new float[t]();
        // 数据是一模一样的，引用指针会不会好一点呢？
        for (int i = 0; i < t; i++) {
            data[i] = L.data[i];
        }
    }
    // 析构函数
    ~Layer() {
        delete[] data;
    }
    // 定位： n,c,i,j都是从0开始的
    //A(n,c,i,j)第n个，第c层，第i行，第j列
    float& operator()(int n, int c, int i, int j) {
        int index = n * ((height) * (weight)*channel) + c * (height) * (weight)+i * (weight)+j;
        return data[index];

    }
    //B(n,c,i,j)第n个，第c层，第i行，第j列
    const float operator()(int n, int c, int i, int j)const {
        int index = n * ((height) * (weight)*channel) + c * (height) * (weight)+i * (weight)+j;
        return data[index];
    }
    //A=B 
    Layer& operator = (const Layer& L) {
        number = L.number;
        channel = L.channel;
        height = L.height;
        weight = L.weight;

        int t = number * channel * height * weight;
        if (data) {
            delete[] data;
        }
        data = new float[t]();
        for (int i = 0; i < t; i++) {
            data[i] = L.data[i];
        }

    }
    //cout<<A
    friend std::ostream& operator << (std::ostream& os, const Layer& L) {
        /*
        for(int i = 0; i < (L.number*L.channel*(L.weight+2*L.pad)*(L.height+2*L.pad));i++){
            if(i>0 && i%(L.weight+2*L.pad)==0){
                os << endl;
            }
            if(i>0 && i%((L.weight+2*L.pad)*(L.height+2*L.pad))==0){
                os << endl;
            }
            os << L.data[i] << " ";

        }
        return os;
        */
        for (int n = 0; n < L.number; n++) {
            os << "number : " << n << endl;
            for (int c = 0; c < L.channel; c++) {
                os << "channel : " << c << endl;
                for (int i = 0; i < L.height; i++) {
                    for (int j = 0; j < L.weight; j++) {
                        os << L(n, c, i, j) << " ";
                    }
                    os << endl;
                }
                os << endl;
            }
            os << endl;
        }
        return os;
    }


    /* --------------- conv： C = A*kernel+bias [ReLU]-------------- */
    // original
    /*
        Layer conv(const Layer & kernel, float * bias, int pad, int stride){
            if(channel == kernel.channel){
                int out_channel = kernel.number;
                int out_height = (height+2*pad-kernel.height)/stride+1;
                int out_weight = (weight+2*pad-kernel.weight)/stride+1;
                float * out_data = new float [1*out_channel*out_height*out_weight]();
                Layer temp(1,out_channel,out_height,out_weight,out_data);
                Layer after_pad = (*this).padding(pad);

                float sum = 0;
                int kernel_total = kernel.height*kernel.weight*kernel.channel;
                int bj(0),bc(0),bi(0),bjk(0),bck(0),bik(0);
                int h(height+2*pad),w(weight+2*pad);

                for(int c = 0; c < out_channel;c++){
                    for(int i = 0; i < out_height ; i++){
                        for(int j = 0; j < out_weight ; j++){
                            //cout << "(" << i << "," << j << "): " ;

                            for(int k_c = 0; k_c < kernel.channel ; k_c++){
                                bc = k_c*h*w;
                                bck = k_c*kernel.height*kernel.weight;
                                for(int k_i = 0; k_i < kernel.height ; k_i++){
                                    bi = k_i*w;
                                    bik = k_i*kernel.weight;
                                    for(int k_j = 0; k_j<kernel.weight ; k_j++){
                                        bj = bc+bi+k_j;
                                        bjk = bck+bik+k_j;
                                        sum += after_pad.data[bj+stride*(i*w+j)]*kernel.data[c*kernel_total + bjk];


                                    //sum += after_pad(0,k_c,stride* i+k_i,stride*j+k_j)*kernel(c,k_c,k_i,k_j);

                                    //sum += after_pad.data[k_c*(height)*(weight)+(stride* i+k_i)*weight+stride*j+k_j]*kernel.data[c*((height)*(weight)*channel) + k_c*(height)*(weight) + k_i*(weight)+k_j]
                                    //(0,k_c,stride* i+k_i,stride*j+k_j)*kernel(c,k_c,k_i,k_j);

                                    //cout << after_pad(0,k_c,stride* i+k_i,stride*j+k_j) <<" ";
                                   // << "*" <<kernel(c,k_c,k_i,k_j) << "+" ;
                                    }
                                }
                            }

                            //cout << " = " << sum << endl;
                            //cout << endl;


                            // ReLu
                            if((sum+bias[c])>0){
                                temp(0,c,i,j)=sum+bias[c];
                            }
                            sum = 0;
                        }
                    }
                }
                return temp;
            }else{
                cout << "sizes don't match! conv failed!" << endl;
                return *this;
            }
        }


    */
    // optimized 1 ： 不使用operator()，直接定位
    /*
        Layer conv(const Layer & kernel, float * bias, int pad, int stride){
            if(channel == kernel.channel){
                int out_channel = kernel.number;
                int out_height = (height+2*pad-kernel.height)/stride+1;
                int out_weight = (weight+2*pad-kernel.weight)/stride+1;
                float * out_data = new float [1*out_channel*out_height*out_weight]();
                Layer temp(1,out_channel,out_height,out_weight,out_data);
                Layer after_pad = (*this).padding(pad);

                float sum = 0;
                int kernel_total = kernel.height*kernel.weight*kernel.channel;
                int bj(0),bc(0),bi(0),bjk(0),bck(0),bik(0);
                int h(height+2*pad),w(weight+2*pad);

                for(int c = 0; c < out_channel;c++){
                    for(int i = 0; i < out_height ; i++){
                        for(int j = 0; j < out_weight ; j++){
                            //cout << "(" << i << "," << j << "): " ;

                            for(int k_c = 0; k_c < kernel.channel ; k_c++){
                                bc = k_c*h*w;
                                bck = k_c*kernel.height*kernel.weight;
                                for(int k_i = 0; k_i < kernel.height ; k_i++){
                                    bi = k_i*w;
                                    bik = k_i*kernel.weight;
                                    for(int k_j = 0; k_j<kernel.weight ; k_j++){
                                        bj = bc+bi+k_j;
                                        bjk = bck+bik+k_j;
                                        sum += after_pad.data[bj+stride*(i*w+j)]*kernel.data[c*kernel_total + bjk];


                                    //sum += after_pad(0,k_c,stride* i+k_i,stride*j+k_j)*kernel(c,k_c,k_i,k_j);

                                    //sum += after_pad.data[k_c*(height)*(weight)+(stride* i+k_i)*weight+stride*j+k_j]*kernel.data[c*((height)*(weight)*channel) + k_c*(height)*(weight) + k_i*(weight)+k_j]
                                    //(0,k_c,stride* i+k_i,stride*j+k_j)*kernel(c,k_c,k_i,k_j);

                                    //cout << after_pad(0,k_c,stride* i+k_i,stride*j+k_j) <<" ";
                                   // << "*" <<kernel(c,k_c,k_i,k_j) << "+" ;
                                    }
                                }
                            }

                            //cout << " = " << sum << endl;
                            //cout << endl;


                            // ReLu
                            if((sum+bias[c])>0){
                                temp(0,c,i,j)=sum+bias[c];
                            }
                            sum = 0;
                        }
                    }
                }
                return temp;
            }else{
                cout << "sizes don't match! conv failed!" << endl;
                return *this;
            }
        }
    */
    // optimized 2 :  在1的基础上，当卷积核的weight=3，height=3是展开最内两层for循环 
    Layer conv(const Layer& kernel, float* bias, int pad, int stride) {
        if (channel == kernel.channel) {
            int out_channel = kernel.number;
            int out_height = (height + 2 * pad - kernel.height) / stride + 1;
            int out_weight = (weight + 2 * pad - kernel.weight) / stride + 1;
            float* out_data = new float[1 * out_channel * out_height * out_weight]();
            Layer temp(1, out_channel, out_height, out_weight, out_data);
            Layer after_pad = (*this).padding(pad);

            float sum = 0;
            float a00, a01, a02, a10, a11, a12, a20, a21, a22;
            a00 = a01 = a02 = a10 = a11 = a12 = a20 = a21 = a22 = 0;
            int a0, a1, a2, k;
            a0 = a1 = a2 = k = 0;
            int WH = after_pad.weight * after_pad.height;
            int SW = stride * after_pad.weight;
            int whc = 9 * kernel.channel;
            int KI(0), KJ(0);
            int kc(0);
            int tc(0), ti(0), tj(0);


            for (int c = 0; c < out_channel; c++) {
                kc = c * whc;
                tc = c * out_height * out_weight;
                for (int i = 0; i < out_height; i++) {
                    KI = i * SW;
                    ti = i * out_weight;
                    for (int j = 0; j < out_weight; j++) {
                        KJ = j * stride;
                        //cout << "(" << i << "," << j << "): " ;
                        // ****************************************************
                        // 在本次project中kernel的height和weight是固定的3*3
                        // 计算可以减少循环的次数
                        for (int k_c = 0; k_c < kernel.channel; k_c++) {
                            /* original
                           a00 = after_pad(0,k_c,stride* i,stride*j)*kernel(c,k_c,0,0);
                           a01 = after_pad(0,k_c,stride* i,stride*j+1)*kernel(c,k_c,0,1);
                           a02 = after_pad(0,k_c,stride* i,stride*j+2)*kernel(c,k_c,0,2);
                           a10 = after_pad(0,k_c,stride* i+1,stride*j)*kernel(c,k_c,1,0);
                           a11 = after_pad(0,k_c,stride* i+1,stride*j+1)*kernel(c,k_c,1,1);
                           a12 = after_pad(0,k_c,stride* i+1,stride*j+2)*kernel(c,k_c,1,2);
                           a20 = after_pad(0,k_c,stride* i+2,stride*j)*kernel(c,k_c,2,0);
                           a21 = after_pad(0,k_c,stride* i+2,stride*j+1)*kernel(c,k_c,2,1);
                           a22 = after_pad(0,k_c,stride* i+2,stride*j+2)*kernel(c,k_c,2,2);
                           */
                           // updated 
                            a0 = k_c * WH + KI + KJ;
                            k = kc + k_c * 9;
                            a00 = after_pad.data[a0] * kernel.data[k];
                            a01 = after_pad.data[a0 + 1] * kernel.data[k + 1];
                            a02 = after_pad.data[a0 + 2] * kernel.data[k + 2];
                            a1 = a0 + after_pad.weight;
                            a10 = after_pad.data[a1] * kernel.data[k + 3];
                            a11 = after_pad.data[a1 + 1] * kernel.data[k + 4];
                            a12 = after_pad.data[a1 + 2] * kernel.data[k + 5];
                            a2 = a0 + 2 * after_pad.weight;
                            a20 = after_pad.data[a2] * kernel.data[k + 6];
                            a21 = after_pad.data[a2 + 1] * kernel.data[k + 7];
                            a22 = after_pad.data[a2 + 2] * kernel.data[k + 8];

                            sum = sum + a00 + a01 + a02 + a10 + a11 + a12 + a20 + a21 + a22;
                        }
                        // ************************************************                
                        // ReLu
                        if ((sum + bias[c]) > 0) {
                            tj = tc + ti + j;
                            //temp(0,c,i,j)=sum+bias[c];
                            temp.data[tj] = sum + bias[c];
                        }
                        sum = 0;
                    }
                }
            }
            return temp;
        }
        else {
            cout << "sizes don't match! conv failed!" << endl;
            return *this;
        }
    }

    /* ----------------------- padding : +0 ------------------------ */
    // original 
    /*
        Layer padding(int pad){
            int h = height + 2*pad;
            int w = weight + 2*pad;
            int t = number * channel * h*w;
            float * d = new float [t]();
            Layer after_pad(number,channel,h,w,d);

            for(int n=0; n<number;n++){
                for(int c=0; c< channel;c++){
                    for(int i=0; i<height;i++){
                        for(int j=0; j<weight;j++){
                            after_pad(n,c,i+pad,j+pad) = (*this)(n,c,i,j);
                        }
                    }
                }
            }
            return after_pad;

        }

    */
    //  optimized ： 不使用operator()，直接定位
    Layer padding(int pad) {
        int h = height + 2 * pad;
        int w = weight + 2 * pad;
        int t = number * channel * h * w;
        float* d = new float[t]();
        Layer after_pad(number, channel, h, w, d);

        int hw = h * w;
        int HW = height * weight;
        int chw = channel * hw;
        int CHW = channel * HW;
        int kn(0), kc(0), ki(0), kj(0), k(0);
        int KN(0), KC(0), KI(0), KJ(0), K(0);

        for (int n = 0; n < number; n++) { // n=0
            kn = n * chw;
            KN = n * CHW;
            for (int c = 0; c < channel; c++) {
                kc = c * hw;
                KC = c * HW;
                for (int i = 0; i < height; i++) {
                    ki = (i + pad) * w + pad;
                    KI = i * weight;
                    kj = kn + kc + ki;
                    KJ = KN + KC + KI;
                    for (int j = 0; j < weight; j++) {
                        after_pad.data[kj + j] = (*this).data[KJ + j];
                    }
                }
            }
        }
        return after_pad;

    }

    /* ------------------------ max pooling ------------------------*/
    Layer maxp() {
        if (weight % 2 == 0 && height % 2 == 0) {
            int N(number), C(channel), H(height / 2), W(weight / 2);
            float* mpp = new float[N * C * H * W]();
            float max(0);
            Layer mp(N, C, H, W, mpp);
            for (int n = 0; n < N; n++) {
                for (int c = 0; c < C; c++) {
                    for (int i = 0; i < H; i++) {
                        for (int j = 0; j < W; j++) {
                            max = (*this)(n, c, 2 * i, 2 * j);
                            if ((*this)(n, c, 2 * i, 2 * j + 1) >= max) {
                                max = (*this)(n, c, 2 * i, 2 * j + 1);
                            }
                            if ((*this)(n, c, 2 * i + 1, 2 * j) >= max) {
                                max = (*this)(n, c, 2 * i + 1, 2 * j);
                            }
                            if ((*this)(n, c, 2 * i + 1, 2 * j + 1) >= max) {
                                max = (*this)(n, c, 2 * i + 1, 2 * j + 1);
                            }
                            mp(n, c, i, j) = max;
                        }
                    }
                }
            }
            return mp;
        }
        else {
            cout << "The height and weight are odd number! Maxpool failed!" << endl;
            return (*this);
        }

    }

    /* --------------------- fully-Connected layer------------------------*/
    // original 
    /*
        float * fc(float* weigh, float * bias,int N){
            float * output = new float[N];
            int L = number*channel*height*weight;
            //cout << channel << "*" << height <<"*" << weight << endl;
            float sum = 0;
            for(int i=0; i<N;i++){
                sum =0;
                for(int j=0; j<L;j++){
                sum += weigh[i*L+j]*data[j];
                //cout << weigh[i*4+j] << " * " << input[j] << " + ";
                }
            //cout << " = " << sum << endl;
            output[i] = sum + bias[i];
        }
        return output;

        }
    };
    */
    // optimized :  如果输入数据L长度为4的倍数，将向量相乘分段
    float* fc(float* weigh, float* bias, int N) {
        float* output = new float[N];
        int L = number * channel * height * weight;
        //cout << channel << "*" << height <<"*" << weight << endl;
        float sum = 0;
        for (int i = 0; i < N; i++) {
            sum = 0;
            if (L % 4 == 0) {
                int L1(L / 4), L2(L / 2), L3(L * 3 / 4);
                for (int j = 0; j < L1; j++) {
                    sum += weigh[i * L + j] * data[j];
                    sum += weigh[i * L + L1 + j] * data[L1 + j];
                    sum += weigh[i * L + L2 + j] * data[L2 + j];
                    sum += weigh[i * L + L3 + j] * data[L3 + j];
                }
            }
            else {
                for (int j = 0; j < L; j++) {
                    sum += weigh[i * L + j] * data[j];
                    //cout << weigh[i*4+j] << " * " << input[j] << " + ";
                }
            }

            //cout << " = " << sum << endl;
            output[i] = sum + bias[i];
        }
        return output;

    }
};




int main() 
{

    /*------------------------------------- step1 : read photo(3*128*128) -------------------------------------*/
	// face & bg
	/*
	Mat image = imread("C:\\Users\\dell\\Desktop\\南科大\\课程资料\\大二上学期\\C++\\project\\project2\\SimpleCNNbyCPP-main\\samples\\face.jpg");
	Mat image = imread("C:\\Users\\dell\\Desktop\\南科大\\课程资料\\大二上学期\\C++\\project\\project2\\SimpleCNNbyCPP-main\\samples\\bg.jpg");
	//imshow("face!", image);
	//waitKey(0);

	// Mat image -> array
	int length = image.rows * image.cols * image.channels();
	int l1(length / 3), l2(2 * l1);
	unsigned char* array = new unsigned char[length];
	if (image.isContinuous())
		array = image.data;


	float* arrf = new float[length];
	float t0(0),t1(0),t2(0);


	// BGR->RGB
	
	for (int i = 0; i < length/3; i++) {
		arrf[i] = (float) array[3 * i + 2]/255;
		arrf[i + l1] = (float) array[3 * i + 1]/255;
		arrf[i + l2] = (float) array[3 * i ]/255;
	}
	
	cout << endl << "------------end!----------" << endl;

	ofstream face(" face_data_04.txt");
	if (face.is_open()) {
		cout << "face_data_04 is opened!" << endl;
		for (int i = 0; i < length; i++) {
			face << arrf[i] << ", ";
		}
		face.close();
	}
	else {
		cout << "Unable to open the file!" << endl;
	}
	
	*/

	// general 
	Mat image = imread("C:\\Users\\dell\\Desktop\\南科大\\课程资料\\大二上学期\\C++\\project\\project2\\SimpleCNNbyCPP-main\\samples\\bg.jpg");

	// Mat image -> array
	int length = image.rows * image.cols * image.channels();
	int l1(length / 3), l2(2 * l1);
	unsigned char* array = new unsigned char[length];
	float* arrf = new float[length];
	if (image.isContinuous()){ array = image.data; }		
	
	// BGR->RGB & normal & char->float

	for (int i = 0; i < length / 3; i++) {
		arrf[i] = (float)array[3 * i + 2] / 255;
		arrf[i + l1] = (float)array[3 * i + 1] / 255;
		arrf[i + l2] = (float)array[3 * i] / 255;
	}
    //cout << arrf[0] << endl;

	// data -> txt
	/* 把数据读到txt里面
	ofstream bg(" bg_data_00.txt");
	if (bg.is_open()) {
		cout << "bg_data_00 is opened!" << endl;
		for (int i = 0; i < length; i++) {
			bg << arrf[i] << ", ";
		}
		bg.close();
	}
	else {
		cout << "Unable to open the file!" << endl;
	}
	*/

    /*------------------------------------------- step2 : cnn ---------------------------------------------------*/


    // test data
    /*
    cout << "test : bg // visual studio // orginal " << endl;
    for (int i = 0; i < 10; i++) {
        auto t1 = std::chrono::steady_clock::now();
        Layer conv0_input(1, 3, 128, 128, arrf);
        Layer conv0_kernel(16, 3, 3, 3, conv0_weight);
        Layer conv1_input = conv0_input.conv(conv0_kernel, conv0_bias, 1, 2).maxp();
        Layer conv1_kernel(32, 16, 3, 3, conv1_weight);
        Layer conv2_input = conv1_input.conv(conv1_kernel, conv1_bias, 0, 1).maxp();
        Layer conv2_kernel(32, 32, 3, 3, conv2_weight);
        Layer conv3_output = conv2_input.conv(conv2_kernel, conv2_bias, 1, 2);

        //------- fc -------
        int N = 2;
        float* scores = new float[2];
        scores = conv3_output.fc(fc0_weight, fc0_bias, N);

        //--- soft max ---
        double p01 = exp(scores[0]);
        double p02 = exp(scores[1]);
        double p1 = p01 / (p01 + p02);
        double p2 = p02 / (p01 + p02);

        auto t2 = std::chrono::steady_clock::now(); //结束时间
        double time = std::chrono::duration<double, std::milli>(t2 - t1).count();

        cout <<time<< endl;

        delete[] scores;
    }
    */

    /*
    * cnn 原始操作 */
    cout << "---- \"bg.jpg\" in visual studio ---- "<< endl;
    //cout << "--- \"face.jpg\" in visual studio --- " << endl;
    auto t1=std::chrono::steady_clock::now();
    Layer conv0_input(1, 3, 128, 128, arrf);
    Layer conv0_kernel(16, 3, 3, 3, conv0_weight);
    Layer conv1_input = conv0_input.conv(conv0_kernel, conv0_bias, 1, 2).maxp();
    Layer conv1_kernel(32, 16, 3, 3, conv1_weight);
    Layer conv2_input = conv1_input.conv(conv1_kernel, conv1_bias, 0, 1).maxp();
    Layer conv2_kernel(32, 32, 3, 3, conv2_weight);
    Layer conv3_output = conv2_input.conv(conv2_kernel, conv2_bias, 1, 2);

    //------- fc -------
    int N = 2;
    float* scores = new float[2];
    scores = conv3_output.fc(fc0_weight, fc0_bias, N);
    //cout << scores[0] << endl;
    //cout << scores[1] << endl;

    //--- soft max ---
    double p01 = exp(scores[0]);
    double p02 = exp(scores[1]);
    double p1 = p01 / (p01 + p02);
    double p2 = p02 / (p01 + p02);

    auto t2 = std::chrono::steady_clock::now(); //结束时间
    double time = std::chrono::duration<double, std::milli>(t2 - t1).count();

    cout << "  bg score  : " << p1 << endl;
    cout << " face score : " << p2 << endl;
    cout << "(time: " << time << "ms)" << endl;
    cout << "-----------------------------------" << endl;
    

    delete[] scores;
    //*/

    //?????????????? delete了竟然报错？？？
    //delete[] array;
    //delete[] arrf;

	return 0;
}

/* location : 
* C:\\Users\\dell\\Desktop\\南科大\\课程资料\\大二上学期\\C++\\project\\project2\\SimpleCNNbyCPP-main\\samples\\
*/