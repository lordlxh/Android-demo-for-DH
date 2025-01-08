# Android-demo-for-DH
本项目是一个安卓程序，用来在安卓端推理Ultralight Digital Human项目，仅供学习参考。

# 本端侧代码可行的使用方式：
1、找个安卓机器安装本文件夹中提供的apk

2、利用pnnx将之前训练出来的pth模型权重文件进行转换。

3、将其中的w.ncnn.bin和w.ncnn.param重命名为db和dp

4、从 https://drive.google.com/file/d/1e4Z9zS053JEWl6Mj3W9Lbc9GDtzHIg6b/view?usp=drive_link 获取encoder.onnx 文件，重命名为wo；

5、将apk打开，联网，进行下载基础模型和人物模型，之后尝试运行，但一定会闪退（正常现象，因为下载的模型是硅基的，后面需要换成我们自己的）；

6、运行jpg2sij.py代码，将之前的训练集转换为安卓需要的sij文件(可手动挑选截取一部分)，将文件夹重命名为raw_jpgs；

7、运行lmstojson.py代码，将之前的训练集转换为安卓需要的文件格式，并且重命名为bj(序号一定要和上面的sij文件对齐)；

8、根据自己需要修改cj，我们这边用了它原有的

9、使用USB调试进入到手机Android；

10、打开手机目录进入Internal storage\Android\data\ai.guiji.duix.test\files\duix

11、进入gj_dh_res,将第4步中的wo文件复制进去

12、进入liangwei_540s，将第3步，第6步，第7步，第8步的文件复制进去，并且删除raw_sg和pha文件夹。

13、点进wav，将你需要推理的语音移动进来重命名为help.wav(注意采样率需要是16000hz，比特率是256k)，并且删除原来的help.wav。

14、重新进入安卓程序，即可进行推理

# 待改进的问题
目前该代码中提取mfcc特征这边由于水平限制，我是强行用硅基源代码的结果去拟合Ultralight Digital Human中fbank的音频提取结果，如果后续有人可以用cpp实现这一部分，可以提交相应的requests！

# 特别感谢！
 [硅基智能SDK](https://github.com/GuijiAI/duix.ai)，[Ultralight Digital Human](https://github.com/anliyuan/Ultralight-Digital-Human)

这两个项目为本项目提供了莫大的帮助。

以及AIMC的[JX_AI](https://github.com/QUTLiJingxiao),[hhh](https://github.com/huang2002)对本项目做出的大量代码贡献。

# 如果本项目对你有帮助，请给我点个star！
