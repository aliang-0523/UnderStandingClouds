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
-- densenet169:0.642(batch_size=4)————kernel效果 densenet169:0.6532(batch_size=10)  
-- densenet201:0.638(batch_size=2)————kernel效果 densenet201:0.649(batch_size=10)  
-- resnet18:(batch_size=)————kernel效果 resnet18:0.640(batch_size=32 accumulate_iter=32)  
-- efficientnetb2:(batch_size=)————kernel效果 efficientnetb2:0.648(batch_size=16)  
-- efficientnetb4:0.6539(batch_size=12)  
-- efficientnetb4:(batch_size=16)  
-- efficientnetb3:(batch_size=)————kernel效果 efficientnetb3:0.651(batch_size=10)(修改了loss为dice_loss以及dice_coef作为metrics)  
-- resnet34_FPN(16 batch) 0.6508  
-- efficientnetb5_FPN(5 batch)0.6567  
-- efficientnetb5_Unet(8_batch)0.6561  
-- efficientnetb7_FPN(4_batch)  0.6575未跑完**ZXY账号**  
-- efficientnetb7_Unet(4_batch)  
-- efficientnetb4_FPN(16_batch)  未跑完**sj账号**
-- densenet169_FPN(10_batch)  
-- efficientnetb5_Unet(5_batch_优化器选用Adam)  0.6509 
## 多分类模型ensemble:
-- 
## 多模型ensemble:

-- se_resnext50_32x4d unet(weight:imagenet)  
-- efficientnet-b5 unet(weight:imagenet)  
-- efficientnet-b5 fpn(weight:imagenet)  
-- efficientnet-b5 fpn(weight:imagenet 训练了较多批次)  
-- se_resnext50_32x4d fpn(weight:imagenet) 0.635(batch-16)  

## ensemble result:
-- post_process threshold:0.32_15000(ds)  
-- post_process threshold:0.30_15000  从0.653提升到0.655  
-- post_process threshold:0.3_13000 0.657  
-- post_process threshold:0.295_13000 0.658(比0.29_13000略好)  
-- post_process threshold:0.2925_13000 0.658(比0.295_13000略好)  
-- post_process threshold:0.29_14000 0.657  
-- post_process threshold:0.29_13000 0.658  
-- post_process threshold:0.29_12000 0.657  
-- post_process threshold:0.28_12000 0.656  
-- post_process threshold:0.28_15000 任然是0.655 比上个稍微差一点  
-- post_process threshold:0.25_12000 0.654  
-- post_process threshold:0.25_10000 0.650  
## ensemble result(efficientnetb4,efficientnetb3,densenet169):
-- post_process threshold:0.2925_13000 0.6572  
-- post_process threshold:0.30_13000 0.6575  
## ensemble result(efficientnetb4_unet,b3_unet,densenet169_unet,densenet121_unet):
-- post_process threshold:0.22_13000 0.6534  
-- post_process threshold:0.2925_13000 0.6587  
-- post_process threshold:0.29_13000  
## ensemble result(efficientnetb3-4_unet,efficientnetb5_FPN,densenet169):
-- post_process threshold:0.2925_13000 0.6582  
-- post_process threshold:0.29_13000 0.6581  
## ensemble result(efficientnetb3_Unet,efficientnetb5_FPN,densenet169_Unet)
-- post_process threshold:0.2925_13000 0.6582  
## ensemble result(efficientnetb4_Unet,efficientnetb5_Unet,efficientnetb5_FPN)  
-- post_process threshold:0.2925_13000 0.6579  
## ensembla result(efficientnetb4_Unet,efficientnetb5_Unet,densenet169_Unet,efficientnetb5_FPN)
-- post_process threshold:0.6_13000 0.6585  
-- post_process threshold:0.5925_13000 0.6594  
-- post_process threshold:0.59_13000 0.6595  
-- post_process threshold:0.5850_13000 0.6593  
-- post_process threshold:0.59_12500 0.6595  
-- post_process threshold:0.45_15000 0.6601  先经过TTA再经过classifier之后效果提升不明显  
-- post_process threshold:0.45_15500 0.6601  先经过TTA再经过classifier之后效果提升不明显  
-- 2019年11月15日10:51:00 将mean ensemble转化为temperature ensemble
-- post_process threshold:0.665_14000 0.611
## ensembla result(efficientnetb4_Unet,efficientnetb5_Unet,densenet169_Unet)  
-- post_process threshold:0.59_13000 0.6573  
## ensembla result(efficientnetb5_Unet,efficientnetb5_FPN,resnet34_FPN,efficientnetb4_Unet)  
-- post_process threshold:0.59_13000 0.6570  
## ensembla result(efficientnetb4_Unet,efficientnetb5_Unet,densenet169_Unet,efficientnetb5_FPN,densenet121_Unet,efficientnetb3_Unet)  
-- post_process threshold:0.6_13000 0.6585  
## ensemble result(efficientnetb4_Unet,efficientnetb7_FPN,efficientnetb5_FPN,efficientnetb5_Unet)  
-- post_process threshold:0.59_13000 0.6593  
## ensemble result(efficientnetb5_Unet,efficientnetb5_FPN,efficientnetb7_FPN)  
--post_process threshold:0.59_13000  0.6573
## ensemble result(efficientnetb5_Unet,efficientnetb5_FPN)
--post_process threshold:0.59_13000  0.6543
## best for now:method  
-- ensemble four models(efficientnetb4_Unet,densenet169_Unet,efficientnetb5_Unet,efficientnetb5_FPN) using temperature mean after post_process threshold 0.665_14000->efficientnetb2 classifier(batch_size=32)(脚本在雄波账号中)  