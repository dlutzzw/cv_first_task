
General statement of the model:<br>
The adopted data set is gesture recognition dataset(The data set comes from 2062 pictures taken in the school competition), and the model is the pre-trained Resnet50 model under pytorch framework.<br>

Introduction to file functions:<br>
gesture_dataset folder--------dataset<br>
test_images folder----------testset<br>
val_dataset.py file---------Classify the dataset into training dataset and validation dataset<br>


Model training steps:<br>
  First of all, because the original dataset is not divided into training dataset and validation dataset,
the val_dataset.py file is run to realize the grouping of dataset，and the results are stored in the train.txt and test.txt files.<br>
  Next, it's time to train our CNN. Run the train_CNN.py file for model training. The final model is stored in the last.pt file<br>
  At last, my model can be applied，set the image_path of the image to be recognized and the model_PATH of the model in the eval.py file,<br>
and run the eval.py file to get the running result.<br>


作品背景：<br>
参加的比赛是卢老师为了锻炼学生动手能力举办的一个校级创新比赛。<br>
创作背景是为了解决：我们学校实验室门禁的人脸识别和指纹识别在疫情期间很不方便的问题，因此突发奇想想要通过手势来进行解锁。<br>
<br>
基本思想：<br>
通过从视频流中取出图像帧，将图像送入CNN分类网络中识别出具体手势，将识别的结果送入txt文件中，每一个手势之间用握拳来区别。<br>
当有四个识别数字输出后和门禁原始密码比对，错误显示erro并且重新识别，最多可尝试3次。<br>
<br>
更多正向、训练、和测试细节见上文General statement of the model！<br>
<br>
反思问题：<br>
由于参加比赛时间较紧，仅仅采用分类算法来实现手势识别问题，整个过程十分完整，包含了深度学习实现任务的各个模块！这也是为什么老师建议从小模型做起的原因。
<br>虽然模型很简单，快速方便部署，但是存在一个很大的问题：就是一张图片仅仅只能出现一只手，如果出现两只以上，则通常会分类错误。
<br><br>
改进思路：<br>
可以将网络改成detection网络，将VOC数据集中训练的一个类别改为hand进行训练，并且根据仅仅输出检测类别为hand的bounding box；
最后根据占整个图像面积最大的手势进行密码识别（面积最大意味着离得最近！）<br>
<br>
实践需要注意：<br>
由于成熟的detection网络比较庞大，参数较多，会存在手势图片样本不够导致样本不均衡的问题，还需想办法解决，例如牺牲准确率减小原有类别样本的数量，图像增强等等。
