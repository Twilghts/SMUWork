import os

from DataManager import DataManager
from Classifier import Classifier



def val(model_dir):
    # 获取目录下所有文件
    files = os.listdir(model_dir)

    # 遍历文件,筛选出后缀为 .pt 的文件
    pt_files = [file for file in files if file.endswith(".pt")]

    # 打印 .pt 文件名
    for file in pt_files:
        # 新建结果文件
        result_path = model_dir + ".txt"
        if not os.path.exists(result_path):
            with open(result_path, "w") as f:
                pass

        data_dir = 'data'
        dataset = DataManager(data_dir)
        model_path = os.path.join(model_dir, file)
        classifier = Classifier(
            dataset, load_model=True, filename=model_path
        )
        with open(result_path, "a") as f:
            print(model_dir.split("/")[-1] + f" epoch={pt_files.index(file) + 1}", file=f)
            classifier.test(on_train_set=False, file=f)



if __name__ == '__main__':
    model_dir = "artifact/checkpoint/lr=0.001 loss=crossentropy"
    val(model_dir)