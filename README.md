# [从卫星图像理解云](https://www.kaggle.com/c/understanding_cloud_organization/overview)
## 评估方法

dice = 2∗|X∩Y|/(|X|+|Y|)

其中x是预测的像素集，y是真值，0<=dice<=1，越大越好

### 编码mask像素
为了减少提交文件的大小，度量对像素值使用长度编码。
例如，“1 3 10 5”表示包含在遮罩中的像素1、2、3、10、11、12、13、14。
检查该度量对是否已排序、为正，并且解码的像素值不重复。像素从上到下编号，然后从左到右编号：1是像素（1,1），2是像素（2,1）等。
### 缩放

预测的编码应该与每边缩放0.25的图像相对应。换言之，虽然训练和测试中的图像为1400 x 2100像素，但预测应缩小到350 x 525像素的图像。为了达到合理的提交评估时间，需要减少提交评估时间。

***********************************************************
## 整理了几个比较好的kernel  
- [使用efficientNetB4进行图片数据训练但是预训练模型未公开,将efficientnet输出转接到unet进行分割任务](https://www.kaggle.com/jpbremer/efficient-net-b4-unet-clouds/notebook)  
- [使用Resnet进行encode,使用unet进行decode(本次提交修复了数据增强包中没有torch.To_Tensor的问题,直接使用作者之前写好的to_tensor)](https://www.kaggle.com/artgor/segmentation-in-pytorch-using-convenient-tools)

***********************************************************
## 都是在bacth_size为16的情况下进行测试
--'vgg11':imagenet','vgg13':'imagenet','vgg16':'imagenet','vgg19':'imagenet','vgg11bn':'imagenet'(未引入),
          'vgg13bn':'imagenet'(未引入),'vgg16bn':'imagenet'(未引入),'vgg19bn'(未引入):'imagenet','densenet121':'imagenet',
          'densenet169':'imagenet','densenet201':'imagenet'(显存不足),'densenet161':'imagenet'(显存不足),'dpn68':'imagenet+5k',
          'dpn68b':'imagenet'(imagenet key error),'dpn92':'imagenet'(imagenet key error),'dpn98':'imagenet+5k'(imagenet+5k key error),'dpn107':'imagenet+5k'(显存不足),'dpn131':'imagenet'(显存不足),
          'inceptionresnetv2':'imagenet'(显存不足),'resnet18':'imagenet'--,'resnet34':'imagenet','resnet50':'imagenet','resnet101':'imagenet',
          'resnet152':'imagenet','se_resnet50':'imagenet','se_resnet101':'imagenet','se_resnet152':'imagenet',
          'se_resnext50_32×4d':'imagenet','se_resnext101_32×4d':'imagenet'  
                    
## 单模型尝试目前进行到densenet161,overfitqueen:dpb68b(当前),dpn68,chaojie xie:vgg16('当前'),vgg19,xixuegui:densenet121,densenet169,slz:vgg11,vgg13,zhou dream:resnet18

**********************************************************
## 本地机实验效果
-- densenet121:0.643(batch_size=4)————kernel效果 densenet121:0.650(batch_size=16)
-- densenet169:0.642(batch_size=4)————kernel效果 densenet169:0.653(batch_size=10)
-- densenet201:0.638(batch_size=2)————kernel效果 densenet201:0.649(batch_size=10)
-- resnet18:(batch_size=)
-- efficientnetb2:0.6(batch_size=)————kernel效果 efficientnetb2:0.648(batch_size=16)