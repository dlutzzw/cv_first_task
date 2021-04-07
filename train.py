import torch            
import torch.nn as nn
from torchvision import transforms      #torchvision主要用于图像的处理和载入
from torch.utils.data import Dataset, DataLoader#？？？？？？？？？？
from PIL import Image
import os                                                          
from torchvision import models
# torch tensorflow(keras) caffe paddlepaddle(baidu)                
EPOCH = 2
BATCH_SIZE = 16
LR = 0.0001
num_classes = 10
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#数据集的设置*****************************************************************************************************************
root ='./gesture_dataset/'    #调用图像

#定义读取文件的格式
def default_loader(path):
    return Image.open(path).convert('RGB')

#首先继承上面的dataset类。然后在__init__()方法中得到图像的路径，然后将图像路径组成一个数组，这样在__getitim__()中就可以直接读取：和下面super有什么关系么
class MyDataset(Dataset): #创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self,txt, transform=None,target_transform=None, loader=default_loader): #*****这里loader说的是加载默认路径内（txt）,加载图片的方式。
        super(MyDataset,self).__init__()#对继承自父类的属性进行初始化
        fh = open(txt, 'r')#按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []
        for line in fh: #迭代该列表#按行循环txt文本中
            
            line = line.rstrip('\n')# 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python？？？？？？？？？？？？？？？
            words = line.split() #用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
            imgs.append((words[0],int(words[1]))) #把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
                                                 # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息列向量////change by zzw ，words[1]是lable       
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader                                    #self作为实例变量实现参数值传递。
        
        #**********以上仅仅更新了imgs******区别于下面的img***********
    def __getitem__(self, index):#这个方法是必须要有的，用于按照索引读取每个元素的具体内容  
        fn, label = self.imgs[index] #当imags是列表形式，imags[index]表示的是第index纵列 #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息fn是图片path
        img = self.loader(fn) # 按照路径读取图片        这里还是RGB形式--->往下就是图像预处理的过程
        if self.transform is not None:
            img = self.transform(img) #数据标签转换为Tensor，且进行归一化的过程，转换为Tensor之后，还需dataloader进行送进去。
            #------------提高上面一句话，所以在dataloader之前就已经将数据转化好成GPU可处理的tensor形式。
        return img,label
    def __len__(self):
        return len(self.imgs)              #长度不对呀，是数据集的两倍了-->列表形式

#*********************************************数据集读取完毕********************************************************************
#图像的初始化操作
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((244,244)),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD)
])
test_transforms = transforms.Compose([
    transforms.Resize((244,244)),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD)
])

#********************************************可将这里以上所有代码当成是函数的提前定义*********change by zzw***********

#数据集加载方式设置
train_data=MyDataset(txt='./train.txt', transform=train_transforms)
test_data = MyDataset(txt='./test.txt', transform=test_transforms)

#然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关（zzw理解:就是给数据打包了）
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)
print('num_of_trainData:', len(train_data))         #向量形式也用len()可以么？？？----和getitem一样，是一种特殊方法，这就相当于调用函数。
print('num_of_testData:', len(test_data))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#************************************这里以下讲的是导入，并且构造网络结构，是一个全连接过程。**************************

cnn = models.resnet50(pretrained=True)                      #导入resnet50预训练模型

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    cnn = nn.DataParallel(cnn)
cnn.fc = nn.Linear(2048, num_classes)                           #cnn.to /fc 分别什么意思-----------送到device中，FC是fully connected。
cnn.to(device)
#*************************************这里以下是优化器和调度器的构造，优化器是每个图片batch样本进行一次改变方向，但是一个epoch改变一次学习率（也可以每个batch更新一次LR）
optimizer = torch.optim.Adam(cnn.parameters(),lr = LR)     #训练所有的参数，其他参数默认的
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)        #根据优化器更新学习率的方法，每个epoch更新一次



loss_func = nn.CrossEntropyLoss()   #多用于多分类问题
Loss = []
Loss_list = []
acc_list = []

for epoch in range(EPOCH):
    for step, (b_x,b_y) in enumerate(train_loader):     #迭代   step相当于i=1开始迭代，返回的是类似于MyDataset[i]，只不过现在是小 batch罢了！！

        b_x,b_y = b_x.to(device),b_y.to(device)         #这b_x，b_y地方应该是一个batch长度。
        cnn.train()
        output = cnn(b_x)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()     #更新过一遍参数后梯度清零，上一次epoch计算出来的梯度去掉。
        loss.backward()
        optimizer.step()
        print("  epoch:",epoch, "   step",step, "      loss:  " , loss.item())
        global_iter_num = epoch * len(train_loader) + step + 1  # 计算当前是从训练开始时的第几步(全局迭代次数)----train_loader和train_data有一些不同,最小单位是batch，而后者最小单位是一个行列表
        if step % (len(train_data)//BATCH_SIZE) == 0 and step != 0:   #??这样不是少了一个不完整的小batch么“>”???#这是在每个epoch下最后一步的batch训练完之后进行验证
            Loss_list.append(loss.item())
            print("starting eval ")
            count = 0
            correct = 0
            with torch.no_grad():
                for steps,(test_x,test_y) in enumerate(test_loader):
                    test_x,test_y = test_x.to(device), test_y.to(device)
                    cnn.eval()
                    test_output = cnn(test_x)

                    pred = test_output.max(1, keepdim=True)[1] # 找到概率最大的下标
                    correct += pred.eq(test_y.view_as(pred)).sum().item()   #比较对不对???????
                    acc = float(correct/len(test_loader.dataset))
                acc_list.append(acc)                                            #这里输出对么？？？？？三个好像都不对。
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.detach().cpu().numpy(), '| test accuracy: %.4f' %  float(correct/len(test_loader.dataset)))
           
    sum_numbers = 0  
    for x in Loss_list:  
        sum_numbers += x  
   
    Loss_list = []
PATH ='./last.pt'
torch.save(cnn.state_dict(), PATH)                                  #把第一个epoch训练的模型参数保存，第二个epoch在此基础上进行训练。
