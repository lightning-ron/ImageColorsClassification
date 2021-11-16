from joblib import load
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

CATS = ['Orange','Violet','Red','Blue','Green','Brown','Black']
IMG_SZ = 100

img_path = "test1.jpg"

img = cv2.imread(img_path)

img_resize=cv2.resize(img,(IMG_SZ,IMG_SZ))

# plt.imshow(img_resize)
# plt.show()

l=[img_resize.flatten()]

model = load('predictor.joblib')

probability=model.predict_proba(l)

for ind,val in enumerate(CATS):
     print(f'{val} = {probability[0][ind]*100}%')

print("The predicted color is : "+CATS[model.predict(l)[0]])

print(model.predict(l))