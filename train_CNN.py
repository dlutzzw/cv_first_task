import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torchvision import models
from torch import nn
from torch import optim
from PIL import Image 
import os 

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 1
LR=0.0001
num_classes=10
batchsize=16
#***********************************首先是数据处理部分，包括txt文本中分割出path和label，图像的处理设置成index返回，为了继承Dataset和enumerate调用**********
def default_load(path):
        return Image.open(path).convert("RGB")


class MyData(Dataset):
      def __init__(self,txt,transform,load=default_load):
        super(MyData,self).__init__()       #错1 ---这里不应该打冒号
        fh = open(txt,'r')                  #这里处理'/n'全错了
        imgs=[]
        for line in fh:
        
            line=line.rstrip('\n')
            words=line.split()
            imgs.append((words[0],int(words[1])))       #这里还得把Word1Word0变成数组形式
            
        self.imgs=imgs
        self.transform=transform
        self.load=default_load
        
      def __getitem__(self,index):             #错2----这里忘记打冒号
        fn,label= self.imgs[index]
        imgs=self.load(fn)
        img = self.transform(imgs)
        
        return img,label
      def __len__(self):
        return len(self.imgs)
        
N_mean=[0.485, 0.456, 0.406]      #这里是常用的默认值     
N_std=[0.229, 0.224, 0.225]
     
train_transform = transforms.Compose([
                  transforms.Resize((224,224)),      #错3 1、transforms.compose应该用逗号组合，忘记了   2、调用类首字母应该大写   3、Resize使用错误，使用函数时应该回头看看
                  transforms.ToTensor(),
                  transforms.Normalize(N_mean,N_std)
        ])
test_transform = transforms.Compose([
                  transforms.Resize((224,224)),
                  transforms.ToTensor(),
                  transforms.Normalize(N_mean,N_std)
        ])

#*****************************实例化，为dataloader做准备*************************
train_data=MyData(txt='./train.txt',transform=train_transform)
test_data=MyData(txt='./test.txt',transform=test_transform)

train_loader = DataLoader(train_data,batch_size=batchsize,shuffle=True)         #错7：内置属性参数大小写要一致
test_loader  = DataLoader(test_data,batch_size=batchsize,shuffle=False)
print("number_of_traindata is :",len(train_data))
print("number_of_testdata is :",len(test_data))

#****************************构造CNN网络，和优化器*****************************
cnn= models.resnet50(pretrained=True)
cnn.fc= nn.Linear(2048,num_classes)                                                 #错6：又是类要大写的问题
cnn.to(device)

optimizer = optim.Adam(cnn.parameters(),lr = LR)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

loss_fn = nn.CrossEntropyLoss()

#**************************开始训练网络，且训练完一个epoch之后就验证一下loss和准确率****************
for epoch in range(EPOCH):
    for step, (b_x,b_y) in enumerate(train_loader):      #错4--for循环没加：   且enumerate和range用法类似，都得有in 
        b_x,b_y = b_x.to(device),b_y.to(device)
        
        cnn.train()
        output = cnn(b_x)
        
        optimizer.zero_grad()                              #错7：这里的函数使用没有按照规范来,此外必须了解验证不需要计算损失函数，计算损失函数使用来梯度下降法的。
        loss=loss_fn(output,b_y) 
        loss.backward()
        optimizer.step()
        
        print("     epoch is:",epoch,"       step is:",step,"        loss is:",loss.item())
        
    #***********************************验证集验证准确率*********************************
    print("start eval..........")
    
    correct = 0
    with torch.no_grad():                              #错5-------with torch.no_grad():      是一个上下文件管理器，正常应该有结构性，所以有冒号
        for step1,(e_x,e_y) in enumerate(test_loader):  #且for循环应该在这一层的下面
            e_x,e_y = e_x.to(device),e_y.to(device)
            
            cnn.eval()                                  #错9：没能加上（），相当于空语句
            out_eval = cnn(e_x)
            
            pred = torch.max(out_eval,1)[1]         #返回的是索引值，也就是预测的label，因为最后设定是10个输出，因此索引值也对应着相应的类
            correct += pred.eq(e_y.view_as(pred)).sum().item()
        
            print("The correct number of step :",step1," is:",correct)
        print ("The total step is:",step1)
        print("Total eval is:",len(test_data))
        accur = float(correct/len(test_data))  
        print("epoch is:",epoch,"loss_eval is:",loss.item(),"accurate is:",accur)
        
PATH ='./last.pt'
torch.save(cnn.state_dict(), PATH)                                  #训练好的模型参数保存下来。
        
        
