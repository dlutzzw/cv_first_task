General statement of the model:<br>
The adopted data set is gesture recognition dataset(The data set comes from 2062 pictures taken in the school competition), and the model is the pre-trained Resnet model under pytorch framework.<br>

Introduction to file functions:<br>
gesture_dataset folder--------dataset<br>
test_images folder----------testset<br>
val_dataset.py file---------Classify the dataset into training dataset and validation dataset<br>


Model training steps:<br>
  First of all, because the original dataset is not divided into training dataset and validation dataset,<br>
the val_dataset.py file is run to realize the grouping of dataset，and the results are stored in the train.txt and test.txt files.<br>
  Next, it's time to train our CNN. Run the train_CNN.py file for model training. The final model is stored in the last.pt file<br>
  At last, my model can be applied，set the image_path of the image to be recognized and the model_PATH of the model in the eval.py file,<br>
and run the eval.py file to get the running result.<br>

