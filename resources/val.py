from DataManager import DataManager
from Classifier import Classifier

data_dir = './data'
dataset = DataManager(data_dir)
images_bchw, y_true = dataset.get_sample_model_input()
classifier = Classifier(
    dataset, load_model=False
)

# classifier.test(on_train_set=True)
classifier.test(on_train_set=False)