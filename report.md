# CS205 C/ C++ Program Design Project 2

**Name**: 简欣瑶

**SID**：11911838

## Part 1 -  Implement



### 1.  图片数据获取和处理: 

####  （1）读取

使用opencv 将需要识别的图片读取，并将数据储存在cv::Mat中。需要注意读入图片的类型和地址

> 代码：（以本地文件夹中的bg.jpg为例）

``` c++
#include<opencv2/opencv.hpp> // 头文件
using namespace cv;
//...
Mat image = imread("C:\\Users\\dell\\Desktop\\南科大\\课程资料\\大二上学期\\C++\\project\\project2\\SimpleCNNbyCPP-main\\samples\\bg.jpg");
```

#### （2）存储

将 cv::Mat 中的数据存入数组

1. 数据类型：unsigned char  （0-255）

   (因为图片的像素点在Mat中是以unsigned char类型储存，所以将其依次存入一个unsigned char类的一维数组中。)

2. 数据数量：rows\*cols\*channels 

   (样例中，rows=cols=128,channels=3);

3. 数据顺序：BGR 

   (即，将每一个像素点的三个通道值，按照BGR存，之后是下一个像素点，像素点和像素点按照先横行再竖行的方式依次存储)

   参考网站：http://songnnn.com/?p=179（内含验证代码）

>  代码：

``` c++
// Mat image -> array
int length = image.rows * image.cols * image.channels();
unsigned char* array = new unsigned char[length];
if (image.isContinuous()){ array = image.data; }
```

#### （3）处理

因为CNN模型需要数据类型为float, 范围在[0.0f,1,0f]，并且像素点通道按照RGB的形式排列，对（2）得到的unsigned  char类型的数组 array进行处理

> 代码

``` c++
int l1(length / 3), l2(2 * l1);
float* arrf = new float[length];
for (int i = 0; i < length / 3; i++) {
	arrf[i] = (float) array[3 * i + 2] / 255;
	arrf[i + l1] = (float)array[3 * i + 1] / 255;
	arrf[i + l2] = (float)array[3 * i] / 255;
}
```

1. 数据类型：通过 ` (float）array[i]` 将 unsigned char 类型元素强制转换为 float 类型

2. 规范化 :  直接除255将范围限制在[0.0f,1,0f]

3. 数据顺序：RGB 

   先按行，再按列，最后在按照RGB的顺序将数据存入一维float数组arrf中，即arrf前1/3的数都是R通道的像素点。

   ![image-20210105195143836](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210105195143836.png)



### 2. cnn 的基本实现: 

**总体过程：**

> 原理

在这次的CNN模型中包括三个卷积层（convolutional layer）和一个全连接层(fully connected layer)。在卷积层中，需要实现卷积（conv），修正线性单元（ReLU，Rectified Linear Unit），最大池化（max pooling）。

![image-20210105191157491](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210105191157491.png)

（图片来源：https://faculty.sustech.edu.cn/wp-content/uploads/2020/11/2021010517301552.pdf)

> 实现

``` c++
int main(){
    //...//
    auto t1=std::chrono::steady_clock::now(); //开始时间
    // ---------conv, ReLU, maxpool----------
    // 输入数据
    Layer conv0_input(1,3,128,128,bg_data); 
    // 第一个卷积核的初始化
    Layer conv0_kernel(16,3,3,3,conv0_weight); 
    // 1st : conv + ReLU + maxpool 得到的结果
    Layer conv1_input = conv0_input.conv(conv0_kernel,conv0_bias,1,2).maxp();
    // 第二个卷积核的初始化
    Layer conv1_kernel(32,16,3,3,conv1_weight);
    // 2nd : conv + ReLU + maxpool 得到的结果
    Layer conv2_input = conv1_input.conv(conv1_kernel,conv1_bias,0,1).maxp();
    // 第三个卷积核的初始化
    Layer conv2_kernel(32,32,3,3,conv2_weight);
    // 3rd : conv + ReLU 得到的结果
    Layer conv3_output = conv2_input.conv(conv2_kernel,conv2_bias,1,2);

    //------- fc -------
    int N = 2;
    float * scores = new float[2];
    scores =  conv3_output.fc(fc0_weight,fc0_bias,N); 

    //--- soft max ---
    double p01 = exp(scores[0]);
    double p02 = exp(scores[1]);
    double p1 = p01/(p01+p02);
    double p2 = p02/(p01+p02);

    auto t2=std::chrono::steady_clock::now(); //结束时间
    double time=std::chrono::duration<double,std::milli>(t2-t1).count();
	
    //结果输出
    cout << " bg score  : " << p1 << endl;
    cout << "face score : " << p2 << endl;
    cout << "(time: " << time << "ms)" << endl;

    delete [] scores;
    return 0;
}
```



**类的构建：** 

将输入数据，卷积核以及输出的数据都归为一种类型，命名为 (Layer)， 并将cnn过程分别作为类的成员函数进行实现

- 参数

```c++
class Layer{
private : 
    int number; // 主要针对卷积核设定的参数，对于输入和输出数据其值都为1
    int channel; // 通道数
    int height;  // 高
    int weight;  // 宽
    float * data; // 按照行、列、通道的顺序依次存放数据
    ...};
```

- 基本函数

```c++
class Layer{ ...
public :
    // 默认构造 ：
    Layer():number(0),channel(0),height(0),weight(0),data(nullptr){};
    // 初始化 ： 设定数量、通道、宽和高
    Layer(int n, int c,int h,int w, const float *d);
    // 拷贝构造
    Layer(const Layer & L);
    // 析构函数
    ~Layer();
    // 定位：A(n,c,i,j)第n个，第c层，第i行，第j列(n,c,i,j都是从0开始的)
    float & operator()(int n, int c,int i,int j);
    const float operator()(int n, int c,int i,int j)const；
    // A=B
    Layer & operator = (const Layer & L)；
    //cout<<A
    friend std::ostream & operator << (std::ostream & os, const Layer & L);
```

- 与cnn实现相关的函数

```c++
class Layer{ ...
public :
    //conv ： C = A*kernel+bias [ReLU]
    Layer conv(const Layer & kernel, float * bias, int pad, int stride);
	Layer padding(int pad);
	//max pooling
	Layer maxp();
	//fully connected layer
    float * fc(float* weigh, float * bias,int N);
```



#### （1）conv & ReLU

**理论基础：**

![image-20210105202637886](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210105202637886.png)

（图片来源：https://faculty.sustech.edu.cn/wp-content/uploads/2020/11/2021010517301552.pdf)

**实现函数：**

> **声明**

- 函数：

```c++
Layer conv(const Layer & kernel, float * bias, int pad, int stride); 
//卷积操作，包括ReLU过程 
//C = A*kernel+bias [ReLU]
```

​	参数说明：

| 参数                 | 解释                                                         |
| -------------------- | ------------------------------------------------------------ |
| const Layer & kernel | 卷积核，同样为Layer类型, <br />其通道数应与输入数据的通道数一致<br />卷积核的个数与输出数据的通道数一致 |
| float * bias         | 数组长度与卷积核的个数相同                                   |
| int pad              | 在输入每一层通道填充0的行列数，根据需要可设定位任意正整数    |
| int stride           | 卷积核移动的步长，影响输出数据的大小                         |

- 函数：

``` c++
Layer padding(int pad); //对输入数据进行处理，在每一层通道的周围加上输入整数的0。
```

> **实现过程：**

> 1. 预处理
>    1. 可行性检验 
>    2. 参数设定
>    3. padding
> 2. 卷积相乘 
>    1. 卷积
>    2. ReLu

**预处理：**

- 可行性检验 : 卷积核通道数应与输入数据的通道数一致

``` c++
if(channel == kernel.channel){ 
    // ... 卷积运算
}else{
    cout << "sizes don't match! conv failed!" << endl;
    return *this;
}
```

- 参数设定

``` C++
// 计算输出数据的相关参数
int out_channel = kernel.number; 
int out_height = (height+2*pad-kernel.height)/stride+1;
int out_weight = (weight+2*pad-kernel.weight)/stride+1;
float * out_data = new float [1*out_channel*out_height*out_weight](); 
// 初始化临时的类，这个类将会作为返回值传出到输出数据类中
Layer temp(1,out_channel,out_height,out_weight,out_data);
```

- padding

``` c++
// 在conv中调用padding函数
Layer after_pad = (*this).padding(pad);
```

函数实现：

``` c++
    Layer padding(int pad){
        int h = height + 2*pad;
        int w = weight + 2*pad;
        int t = number * channel * h * w;
        float * d = new float [t](); //(1)
        Layer after_pad(number,channel,h,w,d);

        for(int n=0; n<number;n++){ // number 对于输入值和输出值都为1
            for(int c=0; c< channel;c++){ 
                for(int i=0; i<height;i++){
                    for(int j=0; j<weight;j++){
                        // channel，height,weight都是padding前的数据
                        after_pad(n,c,i+pad,j+pad) = (*this)(n,c,i,j);  //(2)
                    }
                }
            }    
        }
        return after_pad;        
    }
```

说明：

1. ``` c++
   float * d = new float [t](); //(1)
   ```

d为Padding后类的数组，将其中的数据都初始化为0，这样就只需对原有数组的数进行处理，不需要再补上其他数据

2.  ``` c++ 
   after_pad(n,c,i+pad,j+pad) = (*this)(n,c,i,j); //(2)
    ```

<img src="C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210105212005365.png" alt="image-20210105212005365" style="zoom:67%;" />

（图片来源：[百度](https://image.baidu.com/search/detail?ct=503316480&z=0&ipn=d&word=cnn%20padding&step_word=&hs=0&pn=6&spn=0&di=1020&pi=0&rn=1&tn=baiduimagedetail&is=0%2C0&istype=2&ie=utf-8&oe=utf-8&in=&cl=2&lm=-1&st=-1&cs=2018583125%2C3373998232&os=1179135985%2C742244165&simid=4149042141%2C476003435&adpicid=0&lpn=0&ln=265&fr=&fmq=1609852433936_R&fm=result&ic=&s=undefined&hd=&latest=&copyright=&se=&sme=&tab=0&width=&height=&face=undefined&ist=&jit=&cg=&bdtype=11&oriquery=&objurl=https%3A%2F%2Fgimg2.baidu.com%2Fimage_search%2Fsrc%3Dhttp%3A%2F%2Fp1-tt.byteimg.com%2Forigin%2Fpgc-image%2FSFrY4uNA8G8BQq%3Ffrom%3Dpc%26refer%3Dhttp%3A%2F%2Fp1-tt.byteimg.com%26app%3D2002%26size%3Df9999%2C10000%26q%3Da80%26n%3D0%26g%3D0n%26fmt%3Djpeg%3Fsec%3D1612444499%26t%3Dadd136ee52477bea7734fdf41a704ffa&fromurl=ippr_z2C%24qAzdH3FAzdH3Fks52_z%26e3Btpr7k_z%26e3BgjpAzdH3Fmll9mddnAzdH3Fetjofrwvj-d0nnl0l&gsm=7&rpstart=0&rpnum=0&islist=&querylist=&force=undefined)）

从二维矩阵的角度来看，对第c层的某一个点，假设它在原有通道上（上图中绿色趋于）的坐标为（i,j）（第i行，第j列），则它在padding后的通道（加上外层白色的填充区）的(i+pad,j+pad)，即横行和纵行都增加了pad。将二维数组的坐标转换为一维数组的坐标，既可以完成转换。



**卷积相乘**：

- 卷积

``` c++
// 进行卷积相乘
// 对输出数据进行循环（得到其中的每一个数）
for(int c = 0; c < out_channel;c++){            
    for(int i = 0; i < out_height ; i++){
        for(int j = 0; j < out_weight ; j++){
         // 对卷积核进行循环（对应位置相乘相加）
             for(int k_c = 0; k_c < kernel.channel ; k_c++){
             for(int k_i = 0; k_i < kernel.height ; k_i++){
             for(int k_j = 0; k_j < kernel.weight ; k_j++){
		sum += after_pad(0,k_c,stride* i+k_i,stride*j+k_j)*kernel(c,k_c,k_i,k_j);//(*)
             }}} ... }}}
            return temp;
```

说明：(*)

``` c++
sum += after_pad(0,k_c,stride* i+k_i,stride*j+k_j)*kernel(c,k_c,k_i,k_j);
```

定位依据： 

对于输出数据第c层，第i行第j列的数据，它来自于 输入数据每一层 以第（stride\*i）行（stride\*i）列为左上角，以kernel.weight为宽，kernel.height为长的矩形（下图中为正方形）上的每个点和第c个对应层kernel中每个对应位置相乘的乘积的和。因此，需要对卷积核的channel, weight和height分别循环进行计算得到每一个输出的位点的值，再遍历输出数据的channel, weight和height，得到每一个输出的数据。

（在下图中，pad = 0, stride = 1，左侧input蓝色框中(以及其他层相同位置的数据)与kernel卷积得到右侧output第一层的`-2`,在该图中忽略了加bias的操作）

![image-20210105214249089](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210105214249089.png)

（图片来源：https://www.zybuluo.com/hongchenzimo/note/1086311）

- ReLU & bias

> ReLU 原理

<img src="C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210105220012226.png" alt="image-20210105220012226" style="zoom: 50%;" />

> 代码

``` c++
if((sum+bias[c])>0){ temp(0,c,i,j)=sum+bias[c];}    
```

> 说明

对上述卷积计算之后的结果加上对应层的bias进行ReLU操作，如果结果大于零，则将其赋给输出数据对应位置。注意到，在参数设定中

``` c++
float * out_data = new float [1*out_channel*out_height*out_weight](); 
```

即将所有输出的数据都设为0，那么只要对结果大于零的时候进行赋值处理，当结果小于零时可以不进行操作。



#### （2）Max Pooling

> 原理

对于每一个通道上，每2*2的方格中取最大的数放入对应的结果数据中。

![image-20210105221544050](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210105221544050.png)

（图片来源：[百度](https://image.baidu.com/search/detail?ct=503316480&z=0&ipn=d&word=cnn%20maxpooling&step_word=&hs=0&pn=21&spn=0&di=77780&pi=0&rn=1&tn=baiduimagedetail&is=0%2C0&istype=2&ie=utf-8&oe=utf-8&in=&cl=2&lm=-1&st=-1&cs=2794478009%2C2594953227&os=3116433824%2C3359762595&simid=4190726369%2C699665211&adpicid=0&lpn=0&ln=910&fr=&fmq=1609856083810_R&fm=result&ic=&s=undefined&hd=&latest=&copyright=&se=&sme=&tab=0&width=&height=&face=undefined&ist=&jit=&cg=&bdtype=15&oriquery=&objurl=https%3A%2F%2Fgimg2.baidu.com%2Fimage_search%2Fsrc%3Dhttp%3A%2F%2Fimages2015.cnblogs.com%2Fblog%2F1093303%2F201704%2F1093303-20170430195106397-414671399.jpg%26refer%3Dhttp%3A%2F%2Fimages2015.cnblogs.com%26app%3D2002%26size%3Df9999%2C10000%26q%3Da80%26n%3D0%26g%3D0n%26fmt%3Djpeg%3Fsec%3D1612448102%26t%3D41a5dcebf74bb62302ed3f1965ba3a0e&fromurl=ippr_z2C%24qAzdH3FAzdH3Fooo_z%26e3B4w4tv51j_z%26e3Bv54AzdH3Ftgu5-1jpwts-80bncad_z%26e3Bip4s&gsm=15&rpstart=0&rpnum=0&islist=&querylist=&force=undefined)）

> 实现

调用 ` Layer maxp()`成员函数

``` c++
    Layer maxp(){
        // 判断输入的height和weight是否为偶数
        if( weight%2==0 && height%2==0){
            // 输出数据的参数设置
            int N(number),C(channel),H(height/2),W(weight/2);
            float * mpp = new float[N*C*H*W]();
            float max(0);
            // 创建对象
            Layer mp(N,C,H,W,mpp);
            // max pooling
            for(int n=0;n<N;n++){ // N为1
                for(int c=0;c<C;c++){ // 遍历每一层
                    for(int i=0;i<H;i++){ // 遍历输出的行
                        for(int j=0;j<W;j++){ // 遍历输出列
                            // 2*2 四个数中通过比较取最大值
                            max = (*this)(n,c,2*i,2*j);
                            if((*this)(n,c,2*i,2*j+1)>=max){
                                max = (*this)(n,c,2*i,2*j+1);
                            }
                            if((*this)(n,c,2*i+1,2*j)>=max){
                                max = (*this)(n,c,2*i+1,2*j);
                            }
                            if((*this)(n,c,2*i+1,2*j+1)>=max){
                                max = (*this)(n,c,2*i+1,2*j+1);
                            }
                            mp(n,c,i,j) = max;
                        }
                    }
                }
            }
            return mp;
        }else{
            cout << "The height and weight are odd number! Maxpool failed!" << endl;
            return (*this);
        }
        
    }
```



#### （3）Fully-Connected Layer

> 原理

<img src="C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210105222236199.png" alt="image-20210105222236199" style="zoom:50%;" />

假设输出数据的长度为N,输入数据的长度为L，FC即将输入数据（L*1）和给定权重（N\*L）进行内积操作再加上对应的bias(N\*1) 得到输出的结果(N\*1) 。实质为矩阵相乘，更准确的说是矩阵与向量相乘。在此通过通过调用Layer类的成员函数来实现（本模板中 N=2,L=2048）

> 实现

``` c++
float * fc(float* weigh, float * bias,int N){
        float * output = new float[N];
        int L = number*channel*height*weight;
        float sum = 0;
        for(int i=0; i<N;i++){
            sum =0;
            for(int j=0; j<L;j++){
            sum += weigh[i*L+j]*data[j];  
            }
        output[i] = sum + bias[i];
    }
    return output;
    }
```

#### （5）softmax

> 原理

为了使输出的结果数据范围在[0.0,0.1]之间，需要对结果（一个向量：（x1,x2,...））中的数据进行如下的处理：

<img src="C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20210105223150683.png" alt="image-20210105223150683" style="zoom: 50%;" />

（图片来源：https://faculty.sustech.edu.cn/wp-content/uploads/2020/11/2021010517301552.pdf ）

在上图中，p为输出的向量，x为输入的向量，pi,xi分别为其第i个分量，且在本模型中n=2

> 实现

``` c++
#include <cmath> //指数
    double p01 = exp(scores[0]);
    double p02 = exp(scores[1]);
    double p1 = p01/(p01+p02);
    double p2 = p02/(p01+p02);
```

> 说明

scores[0],scores[1]为输入的x1，x2, 而p1,p2即为最终输出的结果。

## Part 2 - Code

https://github.com/Cateatsthatfish/Project-2

## Part 3 - Result & Verification

### 1. 结果演示：

#### result

**bg.jpg**

|            | my          | sample   | delta       | RE       |
| ---------- | ----------- | -------- | ----------- | -------- |
| bg score   | 1           | 0.999996 | 3.64431E-06 | 0.00036% |
| face score | 3.55695e-07 | 0.000004 | 3.64431E-06 | -        |

**face.jpg**

|            | my          | sample   | delta       | RE      |
| ---------- | ----------- | -------- | ----------- | ------- |
| bg score   | 3.76485e-09 | 0.007086 | 0.007085996 | -       |
| face score | 1           | 0.992914 | 0.007085996 | 0.7137% |

>  表格说明：

>  绝对误差absolute error ： delta = |sample - my|
>
> 相对误差Relative Error ：RE = delta/sample\*100%
>
> 在此因为bg的face score和face的bg score都非常小，所以相对误差均以接近1的数值为基计算

#### screen shot

> bg

- vs code

![img](https://uploader.shimo.im/f/gHwIbViRzCOfjwRd.png!thumbnail)

- visual studio

  ![img](https://uploader.shimo.im/f/iY5K0IXhEcyumqQF.png!thumbnail)

- EAIDK-310

![img](https://uploader.shimo.im/f/x9gjXoCDrzJaelmX.png!thumbnail)

- raspberrypi

![img](https://uploader.shimo.im/f/U0aQ5MhVpHjhnSLQ.png!thumbnail)

> face.jpg

- vs code

![img](https://uploader.shimo.im/f/9vuxkTWCWSntY53Z.png!thumbnail)

- visual studio

  ![img](https://uploader.shimo.im/f/ldSGEeNm89jfTJO9.png!thumbnail)

- EAIDK-310

![img](https://uploader.shimo.im/f/HN4AnPxcSskr6Q85.png!thumbnail)

- raspberrypi

![img](https://uploader.shimo.im/f/uFCei4PjghCDR4yq.png!original)

### 2. 速度与优化：

在上述基本实现的过程中，在vscode 平台上的对两个sample的图片进行测速，平均速度为**99.7**ms.

> 数据截图

![img](https://uploader.shimo.im/f/LLbdNNakcMeCA1nJ.png!original)![img](https://uploader.shimo.im/f/SoQfGaHhGSc5qBTJ.png!original)



#### （1）使用引用指针



> 说明

在上述实现卷积的过程中，至少调用了5次拷贝构造，如果在拷贝构造中使用每一个数据的复制，调用一次需要循环（number\*channel\*height\*weight）次，所以考虑使用引用指针来减少循环从而实现加速。

> 代码

``` c++
class Layer{
private :
    int * refcount;
public :
    Layer():number(0),channel(0),height(0),weight(0),data(nullptr){
        refcount = new int[1];
        *refcount = 1;
    };
    Layer(int n, int c,int h,int w, float * d):number(n),channel(c),height(h),weight(w){
        data = d;
        refcount = new int[1];
        *refcount = 1;
    }；
    //拷贝构造
    Layer(const Layer & L): number(L.number),channel(L.channel),height(L.height),weight(L.weight){
        //cout << "it is called!" << endl;
        data = L.data;
        refcount = L.refcount;
        ++ * refcount;
    }
    // 析构函数
    ~Layer(){
        *refcount = *refcount -1;
        if(*refcount == 0){
            //delete [] data;
            delete [] refcount;
        }         
    }        
    //A=B 
    Layer & operator = (const Layer & L){
        number=L.number;
        channel=L.channel;
        height=L.height;
        weight=L.weight;

        *refcount = * refcount - 1;
        if(*refcount == 0){            
            //delete [] data;            
            delete [] refcount;
        }
        data = L.data;
        refcount = L.refcount;
        *refcount = *refcount + 1;
        return (*this);

    }
```

但是需要注意的一点是，如果数据是以静态数组的形式放在头文件里，`delete [] data` 会报错。

> 结果

平均耗时为93.29822222ms

> 截图

![img](https://uploader.shimo.im/f/SgLGvZKa39OtluPQ.png!original)

#### （2）对conv过程进行优化

##### 1.  循环展开

> 原理

原代码中使用了6个for循环，但是在已知kernel的weight 和 height的情况下，可以将最内层两个循环展开，减少循环次数从而达到加速的效果

> 代码

``` c++
Layer conv(const Layer & kernel, float * bias, int pad, int stride){
    \\...   
            for(int c = 0; c < out_channel;c++){            
                for(int i = 0; i < out_height ; i++){
                    for(int j = 0; j < out_weight ; j++){
                        for(int k_c = 0; k_c < kernel.channel ; k_c++){                            
                           a00 = after_pad(0,k_c,stride* i,stride*j)*kernel(c,k_c,0,0);
                           a01 = after_pad(0,k_c,stride* i,stride*j+1)*kernel(c,k_c,0,1);
                           a02 = after_pad(0,k_c,stride* i,stride*j+2)*kernel(c,k_c,0,2);

                           a10 = after_pad(0,k_c,stride* i+1,stride*j)*kernel(c,k_c,1,0);
                           a11 = after_pad(0,k_c,stride* i+1,stride*j+1)*kernel(c,k_c,1,1);
                           a12 = after_pad(0,k_c,stride* i+1,stride*j+2)*kernel(c,k_c,1,2);

                           a20 = after_pad(0,k_c,stride* i+2,stride*j)*kernel(c,k_c,2,0);
                           a21 = after_pad(0,k_c,stride* i+2,stride*j+1)*kernel(c,k_c,2,1);
                           a22 = after_pad(0,k_c,stride* i+2,stride*j+2)*kernel(c,k_c,2,2); 

                           sum = sum + a00+a01+a02+a10+a11+a12+a20+a21+a22 ;                           
                        } ...
                    }
                }                
            }
```

> 结果

耗时：81.6797ms

> 截图

![img](https://uploader.shimo.im/f/COD9FDxhwnSR5MU2.png!thumbnail)

##### 2.  改变取数据的方式

> 原理

在原代码中，为了书写的直观，调用了operator () 的重构来返回一位数组中的数据，但是在6个循环，每个循环有`out_channel*out_height*out_weight*kernel.height*kernel.weight*kernel.channel`次的情况下，反复的调用一个体量非常小的函数非常地耗时，为此通过直接计算定位可以起到加速的作用

``` c++
// 原本调用的定位函数
//A(n,c,i,j)第n个，第c层，第i行，第j列
float & operator()(int n, int c,int i,int j){
        int index = n*((height)*(weight)*channel) + c*(height)*(weight) + i*(weight)+j;
        return data[index];}
```



> 代码

``` c++
//改进前：
sum += after_pad(0,k_c,stride* i+k_i,stride*j+k_j)*kernel(c,k_c,k_i,k_j);
//改进后：
sum += after_pad.data[bj+stride*(i*w+j)]*kernel.data[c*kernel_total + bjk];
```

> 结果

耗时：47.08305556ms

> 截图

![img](https://uploader.shimo.im/f/4xFRZZj5SBayARdJ.png!thumbnail)

#### （3）对fc过程进行优化

> 原理

将矩阵相乘分解为4个部分分别相乘，通过减少循环来实现加速的效果

> 代码

``` c++
        for(int i=0; i<N;i++){
            sum =0;
                int L1(L/4),L2(L/2),L3(L*3/4);
                for(int j=0; j<L1;j++){
                    sum += weigh[i*L+j]*data[j];
                    sum += weigh[i*L+L1+j]*data[L1+j];
                    sum += weigh[i*L+L2+j]*data[L2+j];
                    sum += weigh[i*L+L3+j]*data[L3+j];}
```

> 结果

耗时：108.9983556ms

> 截图

![img](https://uploader.shimo.im/f/2Uqb6YnYy5txqRN5.png!thumbnail)

#### （4）o3加速

> 代码

``` c++
#pragma GCC optimize(3, "Ofast", "inline") 
```

> 结果

耗时：12.950 ms

> 截图

![img](https://uploader.shimo.im/f/EBcpso3KDD0wOLWo.png!thumbnail)



### 3. 平台比较：

在不同平台进行比较的时候，发现平台对速度的影响较大。（以下数据为face.jpg和bg.jpg的多次测试的平均值）

> 数据统计（单位ms）

|          | visual studio | vscode    | EAIDK-310 |
| -------- | ------------- | --------- | --------- |
| 不做加速 | 383.00725     | 99.65603  | 976.0495  |
| o3加速   | 724.3704      | 12.949845 | 41.33033  |

> 耗时比（以visual studio为基础）

|          | visual studio | vscode | EAIDK-310 |
| -------- | ------------- | ------ | --------- |
| 不做加速 | 1             | 0.260  | 2.548     |
| o3加速   | 1             | 0.018  | 0.057     |

​	通过上述几组数据的比较，可以大概发现在没有加速的情况下，vscode 耗时最少，EAIDK-310耗时最多。而在使用o3加速的情况下，仍然是vscode 耗时最少,EAIDK的耗时显著减少，visual studio的耗时没有减少反而增加了。

> o3 对不同平台加速效率的比较

|                        | visual studio | vscode | EAIDK-310 |
| ---------------------- | ------------- | ------ | --------- |
| o3/original （耗时比） | 1.891         | 0.130  | 0.042     |
| o3/original (速度比)   | 0.530         | 7.693  | 23.685    |

由上述数据可以发现，o3 对 EAIDK-310 加速效率最高，可以加速将近20倍，对vscode也有比较明显的加速效果，但是对于visual studio而言，速度反而变成了原来的一半。

> 数据截图

- visual studio 

![img](https://uploader.shimo.im/f/ZadmkQSHB6govS4I.png!thumbnail)![img](https://uploader.shimo.im/f/ucqC7gJMFVxCXdRD.png!thumbnail)![img](https://uploader.shimo.im/f/yxhogGFn9OEY0gdB.png!thumbnail)

- **face // vscode // orginal**

  ![img](https://uploader.shimo.im/f/LLbdNNakcMeCA1nJ.png!thumbnail)![img](https://uploader.shimo.im/f/EBcpso3KDD0wOLWo.png!thumbnail)![img](https://uploader.shimo.im/f/SoQfGaHhGSc5qBTJ.png!thumbnail)



- **EAIDK-310**

<img src="https://uploader.shimo.im/f/WZQ0yuZK5wlSpDV0.png!thumbnail" alt="img" style="zoom:50%;" />

- **raspberrypi** 

<img src="https://uploader.shimo.im/f/MuJXYzN6WNhDNstW.png!thumbnail" alt="img" style="zoom:50%;" />