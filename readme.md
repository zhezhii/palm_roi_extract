***简单对所做的工作进行介绍***

## 1 项目背景
我们在*data/base*文件目录中能够找到2组，从不同设备中提取到的手掌信息。
相同文件夹名意味着相同的手掌，而不同文件夹名意味着不同的手掌。

## 2 项目目标
我们的目标是，通过对这些手掌的信息进行分析，得到手掌较为优秀的区域的静脉特征信息。\
于是我们在palm_roi_extract.py中，对手掌的信息进行了提取，得到了ROI区域的信息。
在得到该ROI区域后，我们在gabor_filter.py中，对ROI区域进行了Gabor滤波，
对特征进行了一定程度的增强。\
当然，上文所说的都是相关的依赖函数，而主体函数是在preprocess.py中进行的。
毕竟这些都只是预处理过程，目的是得到手掌的静脉特征信息。\
而在deepfeature.py中，我们使用了深度学习的方法，对手掌的静脉特征进行了提取。
目的是将来自不同手掌的静脉特征作为特征值，进而实现对手掌的识别分类。

换言之，我们的目标是，通过对手掌的静脉特征进行提取，利用深度学习的方法，对手掌进行识别分类。

## 3 项目函数的介绍
### 1.1 palm_roi_extract.py

function 1:
```python
def evaluate_pic_quality(img, desp="图像"):
    """
    评估图像质量
    :param img: 图像
    :param desp: 图像路径 作为判别到达是哪个图像质量不够好
    :return: 若图像质量不够好, 则返回None, 否则返回处理后的图像和对应的ROI
    """
```
正如注释所言，我们需要传入一个较为优秀的手掌图像，然后对其进行评估。结果将会返回对应的手掌mask和mask区域中的手掌图像。

function 2:
```python
def hand_valley_extract(mask, L=0.2, limit=50):
    """
    提取手掌的两个关键谷点
    :param mask: 原图像轮廓
    :param L: 下端到轮廓中心占图像高度的比例
    :param limit: 边界约束
    :return: 谷点1, 谷点2
    """
```
在这个函数中，我们传入一个手掌mask，然后对其进行处理，得到手掌的两个关键谷点。
这两个谷点，也就是传统意义中的点1（小拇指与无名指之间的谷底）与点2（食指与中指之间的谷底）。
参数L 是为了得到一个合适的截取点， 从而得到一个合适的手腕范围。
参数limit 是为了得到一个合适的截取点， 从而得到一个合适的谷点，一般不作为参数传入。

function 3:
```python
def rdf_min(img, left, right, pref):
    """
    模拟RDF函数
    :param img: mask图像
    :param left: 左侧边界点
    :param right: 右侧边界点
    :param pref: 参考点
    :return: 两边界点之间轨迹到参考点的距离最小点
    """
```
这个函数我们是不必关注的，因为没有合适的现成的函数，于是我才手搓了一个用于function2计算谷点的函数。

但是也还是简单介绍一下，这个函数是为了模拟RDF函数，
从左侧到右侧，依次计算轮廓与参考点之间的距离，然后返回全部的谷点值。
是的，他返回的是一个list，里面记录着全部的谷点值。
（当然，由于是手搓的，所以效率不高，我在function2里面又多做了一些约束，以得到合适的结果）

function 4:
```python
def palm_roi_extract(img, desp="图像"):
    """
    :param img: 手掌图像
    :param mask: 手掌mask图像
    :return: 旋转之后的图像, 抽取的手掌ROI, mask
    """
```

这个其实就是手掌静脉提取的主函数了。主体功能全是包含在里面的。
如果你想得到全部的手掌，谷点，旋转角度，mask，ROI等等，那么你就可以调用这个函数。
甚至重构这个函数。总之，全部的功能都在这里了。

输入图像，依次调用上面的一些函数（嵌套等），
最终得到旋转之后的手掌图像(含mask)，手掌ROI(旋转后)，mask。

实际上，我们后续用到的也都是这个roi，而不是原图像。

### 1.2 gabor_filter.py
function 1:
```python
def build_filters(ksize, sigma, theta, lambd, gamma):
    """
    构建Gabor滤波器
    :param ksize: 滤波器大小
    :param sigma: 高斯函数的标准差
    :param theta: 方向
    :param lambd: 波长
    :param gamma: 空间纵横比
    :return: 滤波器
    """
```
核心功能就是构造Gabor滤波器，这个函数是从网上找的，我也不知道是谁写的了。

function 2:
```python
def process(img, filters):
    """
    Gabor滤波器处理
    :param img: 图像
    :param filters: 滤波器
    :return: 滤波结果
    """
```
这个函数是用来处理图像的，输入图像和滤波器，然后返回滤波结果。

function 3:
```python
def getGabor(img, filters):
    """
    Gabor特征提取
    :param img: 图像
    :param filters: 滤波器
    :return: 滤波结果
    """
```
这个函数是用来提取Gabor特征的，输入图像和滤波器，然后返回滤波结果。

这个是最为核心的函数，也是我们后续用到的函数。我们所进行的Gabor滤波，都是通过这个函数来实现的。
如果想自己手搓，以及找到合适的参数，那么可以调用这个函数。
直接把注释打开，然后重构，慢慢试试就差不多能找到，无论如何，我的参数是对我的图像表现很优秀的。
对于其他图像，就不敢保证了。

function 4:
```python
def auto_threshold(img):
    """
    自动阈值分割
    :param img: 图像
    :return: 二值化图像
    """
```
这个函数是用来自动阈值分割的，输入图像，然后返回二值化图像。
闲着无聊 写作一个函数，其实也就是调用了一下opencv的函数。

### 1.3 preprocess.py

这就是我们在预处理过程中的主函数了，所有的预处理过程都在这里了。
如果不关注后续的分类细节，就只需要看这个函数，然后结合上两个文件中的函数，就能得到全部的预处理过程了。

实际上这里也没有函数，毕竟是主入口。

主要的逻辑就是，
从*data/base*文件夹中，读取全部的手掌图像，
然后对每一张图像进行预处理，得到ROI区域的静脉特征信息，
然后将这些信息保存到*data/test*文件夹或是*data/train*中。

### 1.4 deepfeature.py

主要就是做深度学习的特征提取，以及分类识别，这个实在没必要介绍了，毕竟是调用的函数。

# 如果不理解可以与我联系
# qq: 1245496075
# email: 1245496075@qq.com