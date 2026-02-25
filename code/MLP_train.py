import joblib
import openpyxl
import pandas as pd
import numpy as np
import psutil
from keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.models import load_model, model_from_json
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import os
import time
import datetime
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

tf.random.set_seed(42)

train_path = '../.xlsx'
train_filename = str(train_path.split('/')[-1])

model_dir = f"./modelAndResult/{current_time}"
EPOCHS = 10

feature_range0 = 1
feature_range = 5 #


data = pd.read_excel(train_path)

X = data.iloc[:, feature_range0 :feature_range].values
y = data.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


y_train_categorical = to_categorical(y_train, num_classes=2)
y_test_categorical = to_categorical(y_test, num_classes=2)


feature_size = X_train.shape[1]

def baseline_model():
    model = Sequential()
    model.add(Dense(12, input_dim=feature_size, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    print(model.summary())
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    return model

def train():

    pid = os.getpid()
    p = psutil.Process(pid)


    start_memory = p.memory_info().rss

    estimator = KerasClassifier(build_fn=baseline_model, epochs=EPOCHS, batch_size=1, verbose=1)
    H = estimator.fit(X_train, y_train_categorical, validation_data=(X_test, y_test_categorical), shuffle=True)


    end_memory = p.memory_info().rss


    memory_cost = end_memory - start_memory
    print(f"Memory cost for training: {memory_cost / (1024 * 1024)} MB")


    model_json = estimator.model.to_json()


    os.makedirs(model_dir + "/model", exist_ok=True)


    with open(f"{model_dir}/model/model_test.json", 'w') as json_file:
        json_file.write(model_json)

    estimator.model.save_weights(f'{model_dir}/model/model_test.h5')

    joblib.dump(scaler, f'{model_dir}/model/scaler.save')
    print("……")


    y_pred = estimator.predict(X_test)
    y_val_classes = y_test_categorical.argmax(axis=1)  #


    tn, fp, fn, tp = confusion_matrix(y_val_classes, y_pred).ravel()
    f1 = f1_score(y_val_classes, y_pred, average='macro')

    #
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"F1 Score: {f1}")

    #
    #
    train_loss_values = H.history['loss']  #
    val_loss_values = H.history['val_loss']  #
    train_acc_values = H.history['binary_accuracy']
    val_acc_values = H.history['val_binary_accuracy']
    #
    final_train_loss = train_loss_values[-1]  #
    final_val_loss = val_loss_values[-1]  #
    final_train_acc = train_acc_values[-1]
    final_val_acc = val_acc_values[-1]
    txt_path = f"./modelAndResult/{current_time}/conclusion_test.txt"
    with open(txt_path, "a") as text_file:
        text_file.write(
            f"{current_time}\nName: {train_filename}\ntrainLoss:{final_train_loss}\nValLoss:{final_val_loss}\n"
            f"TrainAcc:{final_train_acc}\nValAcc:{final_val_acc}\n"
            f"TP:{tp}\nFP:{fp}\nTN:{tn}\nFN:{fn}\nF1:{f1}\n"
            f"EPOCHS:{EPOCHS}\nfeature size:{feature_size}\n----------------------\n")

    #
    #
    os.makedirs(model_dir + "/plot", exist_ok=True)

    N = np.arange(0, EPOCHS)

    plt.figure(1)
    plt.style.use("ggplot")
    plt.ylim(0, 10)
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")  #
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()
    #
    plt.savefig(f"./modelAndResult/{current_time}/plot/Loss.jpg")
    print("……")

    plt.figure(2)
    plt.style.use("ggplot")
    plt.ylim(0.6, 1)
    plt.plot(N, H.history["binary_accuracy"], label="train_accuracy")
    plt.plot(N, H.history["val_binary_accuracy"], label="val_accuracy")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend()
    #
    plt.savefig(f"./modelAndResult/{current_time}/plot/Accuracy.jpg")
    print("……")
    plt.show()

    return EPOCHS, H

def test():
    print("=================================================================")
    print("……")
    f_path = f'./testData'
    file = os.listdir(f_path)
    name = []
    tp = []
    tn = []
    fp = []
    fn = []
    acc = []
    f1 = []
    pre = []
    rec = []

    for file_i in file:
        data_path = f_path + "/" + file_i

        #
        test_data = pd.read_excel(data_path)
        X_test = test_data.iloc[:,feature_range0 :feature_range].values #
        y_test = test_data.iloc[:, -1].values #

        #
        #
        scaler = joblib.load(f'{model_dir}/model/scaler.save')
        X_test = scaler.transform(X_test)

        # ========================
        y_test_categorical = to_categorical(y_test, num_classes=2)
        name.append(file_i.replace(".xlsx", ''))

        #
        json_file = open(f"{model_dir}/model/model_test.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(f"{model_dir}/model/model_test.h5")
        loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

        #
        data = X_test.reshape(-1, feature_size)  #
        predicted = loaded_model.predict(data)  #
        predicted_label = np.argmax(predicted, axis=-1)

        #
        true_value = []
        for label in y_test_categorical:
            if str(label) == '[0. 1.]':
                true_value.append(1)
            else:
                true_value.append(0)

        #
        print(f"{file_i}……")
        confusion = confusion_matrix(true_value, predicted_label, labels=[0, 1])
        #
        TN = confusion[0][0]
        FP = confusion[0][1]
        FN = confusion[1][0]
        TP = confusion[1][1]
        #
        TN1 = TN / (TN + FP)
        FP1 = FP / (TN + FP)
        FN1 = FN / (FN + TP)
        TP1 = TP / (FN + TP)
        # tn.append(TN)
        # fp.append(FP)
        # fn.append(FN)
        # tp.append(TP)
        tn.append(TN1)
        fp.append(FP1)
        fn.append(FN1)
        tp.append(TP1)
        # print('TN:', TN)
        # print('FP:', FP)
        # print('FN:', FN)
        # print('TP:', TP)

        precision = TP / (TP + FP)
        pre.append('{:.4f}'.format(precision))
        recall = TP / (TP + FN)
        rec.append('{:.4f}'.format(recall))
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        acc.append('{:.4f}'.format(accuracy))
        F1 = 2 * precision * recall / (precision + recall)
        f1.append('{:.4f}'.format(F1))

    #
    name = np.array(name)
    name = name.reshape(-1, 1)
    tp = np.array(tp)
    tp = tp.reshape(-1, 1)
    tn = np.array(tn)
    tn = tn.reshape(-1, 1)
    fp = np.array(fp)
    fp = fp.reshape(-1, 1)
    fn = np.array(fn)
    fn = fn.reshape(-1, 1)
    pre = np.array(pre)
    pre = pre.reshape(-1, 1)
    rec = np.array(rec)
    rec = rec.reshape(-1, 1)
    acc = np.array(acc)
    acc = acc.reshape(-1, 1)
    print(f"acc:{acc}")
    f1 = np.array(f1)
    f1 = f1.reshape(-1, 1)

    data = np.concatenate((name, tp, fn, tn, fp, pre, rec, f1, acc), axis=1)
    dataFrame = pd.DataFrame(data, columns=['Name', 'TP', 'FN', 'TN', 'FP',
                                            'Precision', 'Recall', 'F1', 'Accuracy'])

    #
    os.makedirs(model_dir + "/result", exist_ok=True)

    mypath = f"./modelAndResult/{current_time}/result/result_test_{current_time}.xlsx"
    #
    mybook = openpyxl.Workbook()
    #
    mybook.save(mypath)

    with pd.ExcelWriter(mypath) as writer:
        dataFrame.to_excel(writer, sheet_name='page1', float_format='%.6f', index=False)
    print(f"{file_i}")


if __name__ == '__main__':
    begin = time.time()
    #
    train()
    #
    test()
    end = time.time()
    time = end - begin
    print(f"Time: {time}")