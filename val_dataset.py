#适用于只含有训练集（需要分割验证集）的分类数据集
import os

def path2label(path,txt_path,file_train,file_test):
        dirs = os.listdir(path)
        dirs.sort()
        for i in range (len(dirs)):   #类别数,与训练和测试文件无关，只是给个范围。dirs[i]是类别名，也就是类别文件夹的名字 , list
            jpg_name = os.listdir(os.path.join(path,dirs[i]))       # 每个类别下图片的名字，str
            jpg_name.sort()
            for m in range (len(jpg_name)):
                path_jpg = os.path.join(path,str(dirs[i]),jpg_name[m]) +' ' + str(int(dirs[i][0:3]))  + '\n'  #图片的绝对路径
                if m%10 == 0:
                   file_test.write(path_jpg)
                else:
                    file_train.write(path_jpg)

#/代表根目录  ./代表当前目录  .../代表上层目录
train_path = './gesture_dataset/' 

train_txt_path = './train.txt'
test_txt_path ='./test.txt'

file_train = open(train_txt_path, 'w')
file_test = open(test_txt_path,'w')

path2label(train_path,train_txt_path,file_train,file_test)
