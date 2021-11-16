import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import os

from tqdm import tqdm

from joblib import dump,load

for dirname, _, filenames in os.walk('/work/archive/ColorClassification'):
    for filename in filenames:
        os.path.join(dirname,filename)

DATA_DIR = '/work/archive/ColorClassification'
CATS = ['Orange','Violet','Red','Blue','Green','Brown','Black']
IMG_SZ = 100

trn_data = []
def create_training_data():
    for cat in CATS:
        path=os.path.join(DATA_DIR,cat)
        class_num=CATS.index(cat)
        for img in os.listdir(path):
            try:
                img_arr=cv2.imread(os.path.join(path,img))
                new_arr=cv2.resize(img_arr,(IMG_SZ,IMG_SZ))
                trn_data.append([new_arr,class_num])
            except Exception as e:
                pass
create_training_data()

setlen = len(trn_data)
print(setlen)

X=[]
y=[]

for CATS, label in trn_data:
    X.append(CATS)
    y.append(label)
X = np.array(X).reshape(setlen,-1)

X.shape

X=X/255.0

y=np.array(y)

y.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y)


from sklearn.svm import SVC
svc = SVC(kernel='linear',gamma='auto',probability=True)
svc.fit(X_train,y_train)

y2=svc.predict(X_test)

from sklearn.metrics import accuracy_score
print( 'Accuracy is', accuracy_score(y_test,y2))

from sklearn.metrics import classification_report
print('Classification Report',classification_report(y_test,y2))

result = pd.DataFrame({'original' : y_test, 'predicted' : y2})

print(result)

dump(svc,'predictor.joblib')