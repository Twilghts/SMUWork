from DataManager import DataManager
from Classifier import Classifier


data_dir = './data'
dataset = DataManager(data_dir)
# dataset.show_sample_images()
images_bchw, y_true = dataset.get_sample_model_input()
# print(images_bchw, images_bchw.shape)
# print(y_true, y_true.shape)
classifier = Classifier(
    dataset, load_model=False
)
classifier.train(lr=1e-3, epochs=20)
#
# classifier.test(on_train_set=True)
# classifier.test(on_train_set=False)