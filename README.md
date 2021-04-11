General statement of the model:
The adopted data set is gesture recognition dataset, and the model is the pre-trained Resnet model under pytorch framework.


Introduction to file functions:
gesture_dataset folder--------dataset
test_images folder----------testset
val_dataset.py file---------Classify the dataset into training dataset and validation dataset


Model training steps:
  First of all, because the original dataset is not divided into training dataset and validation dataset,
the val_dataset.py file is run to realize the grouping of dataset，and the results are stored in the train.txt and test.txt files.
  Next, it's time to train our CNN. Run the train_CNN.py file for model training. The final model is stored in the last.pt file
  At last, my model can be applied，set the image_path of the image to be recognized and the model_PATH of the model in the eval.py file,
and run the eval.py file to get the running result.

