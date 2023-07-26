# recognize-handwritten-Thai-numbers

## Objective
โปรเจ็คนี้เป็นโปรเจ็คที่จัดทำขึ้นเพื่อส่งในวิชา Applied Machine Learning ของสถาบันบัณฑิตพัฒนบริหารศาสตร์(NIDA) โดยวัตถุประสงค์ของโปรเจ็คนี้ คือการสร้าง Machine Learning Model เพื่อจดจำและทำนายตัวเลขภาษาไทยที่เขียนด้วยลายมือ

## 1. Data Collection & Import Python Packages

### 1.1 Data Collection
เริ่มต้น เราได้ทำการเก็บรวบรวมไฟล์รูปภาพที่มีการเขียนตัวเลขภาษาไทยตั้งแต่เลข 0 ถึงเลข 9 ไว้ภายในรูป โดยแต่ละรูปจะมีขนาดเท่ากับ 28x28 pixel
และมีจำนวนตัวเลขละ 40 รูป รวมแล้วเราจะได้ไฟล์รูปภาพที่มีการเขียนตัวเลขภาษาไทยทั้งหมด 400 รูป
ซึ่งมีตัวอย่างลักษณะของตัวเลขภาษาไทยดังนี้

![image](https://github.com/MeenWhile/recognize-handwritten-Thai-numbers/assets/125643589/d9219534-efa0-45cc-b925-795614c4b78f)

### 1.2 Import Python Packages
ต่อมา เราได้ import library ที่จำเป็นในการวิเคราะห์ข้อมูลซึ่งประกอบด้วย
  1. NumPy
  2. Pandas
  3. Matplotlib
  4. csv
  5. opencv
  6. plotly
  7. pycaret
  8. sklearn

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import cv2
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from pycaret.classification import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score,roc_auc_score
```

## 2. Data cleansing

### 2.1 Reposition and Resizing Image
สำหรับการ cleansing data เราได้ทำการปรับให้ตัวเลขภาษาไทยในแต่ละรูปอยู่กึ่งกลางภาพมากขึ้น และขยายตัวเลขให้เต็ม pixel เพื่อให้ ML model สามารถจดจำและทำนายได้อย่างมีประสิทธิภาพ โดยตัวอย่างผลลัพธ์หลังการ cleansing data เป็นดังนี้

Image before cleansing
![image](https://github.com/MeenWhile/recognize-handwritten-Thai-numbers/assets/125643589/43bab274-e230-4202-b6ff-deee6360171d)

Image  after cleansing
![image](https://github.com/MeenWhile/recognize-handwritten-Thai-numbers/assets/125643589/eb470d1b-91ad-46f9-901f-0de0a0e07c4b)

จากนั้นเราก็ได้นำ data หลัง cleansing ทั้งหมดมาแปลงเป็น dataFrame ดังนี้

```python
#For Reposition and Resizing Image

def Count_Up(img,img_check):
    count_up = 0
    check_up = False
    for i in range(28):
        for j in range(28):
            if img[i,j] < max(img_check):
                check_up = True
                break
        if check_up == True:
            break
        count_up += 1
    return count_up

def Count_Down(img,img_check):
    count_down = 0
    check_down = False
    for i in range(27,0,-1):
        for j in range(28):
            if img[i,j] < max(img_check):
                check_down = True
                break
        if check_down == True:
            break
        count_down += 1
    return count_down

def Count_Left(img,img_check):
    count_left = 0
    check_left = False
    for i in range(28):
        for j in range(28):
            if img[j,i] < max(img_check):
                check_left = True
                break
        if check_left == True:
            break
        count_left += 1
    return count_left

def Count_Right(img,img_check):
    count_right = 0
    check_right = False
    for i in range(27,0,-1):
        for j in range(27,0,-1):
            if img[j,i] < max(img_check):
                check_right = True
                break
        if check_right == True:
            break
        count_right += 1
    return count_right

def change_position(img,img_check):
    crop_img = img
    dummy = np.full(28,255,dtype='uint8')
    dummy2 = np.full([28,1],255,dtype='uint8')
    
    count_array = []
    count_array.append(Count_Up(crop_img,img_check))
    count_array.append(Count_Down(crop_img,img_check))
    count_array.append(Count_Left(crop_img,img_check))
    count_array.append(Count_Right(crop_img,img_check))
    
    #vertical_change_position
    avg_vertical = (count_array[0] + count_array[1])/2
    while Count_Up(crop_img,img_check) - avg_vertical > 0.5:
        crop_img = crop_img[1:crop_img.shape[0], :]
        crop_img = np.vstack([crop_img,dummy])
    while Count_Down(crop_img,img_check) - avg_vertical > 0.5:
        crop_img = crop_img[0:crop_img.shape[0]-1, :]
        crop_img = np.vstack([dummy,crop_img])
    
    #horizontal_change_position
    avg_horizontal = (count_array[2] + count_array[3])/2
    while Count_Left(crop_img,img_check) - avg_horizontal > 0.5:
        crop_img = crop_img[:,1:crop_img.shape[1]]
        crop_img = np.hstack([crop_img,dummy2])
    while Count_Right(crop_img,img_check) - avg_horizontal > 0.5:
        crop_img = crop_img[:, 0:crop_img.shape[1]-1]
        crop_img = np.hstack([dummy2,crop_img])

    return crop_img
```

```python
#For Generate CSV and dataFrame

with open("number_All.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    lst_columns = ['y']
    lst_number = []
    for l in range(1,28*28+1):
        lst_columns.append('pixel' + str(l))
    writer.writerow(lst_columns)
    for j in range(0,10):
        for i in range(1,41):
            count_array = []
            file = str(j) + "-" + str(i) + ".png"
            img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
            #img_check = img.flatten()
            img_check = [0,240]
            crop_img = change_position(img,img_check)
            count_array.append(Count_Up(crop_img,img_check))
            count_array.append(Count_Down(crop_img,img_check))
            count_array.append(Count_Left(crop_img,img_check))
            count_array.append(Count_Right(crop_img,img_check))

            crop_img = crop_img[min(count_array):crop_img.shape[0], :]
            crop_img = crop_img[0:crop_img.shape[0]-min(count_array), :]
            crop_img = crop_img[:, min(count_array):crop_img.shape[1]]
            crop_img = crop_img[:, 0:crop_img.shape[1]-min(count_array)]
            crop_img = cv2.resize(crop_img,(28,28))

            crop_img_flatten = crop_img.flatten()
            lst = [j]
            for k in crop_img_flatten:
                lst.append(k)
            writer.writerow(lst)
            lst_number.append(lst)

df_num = pd.DataFrame(lst_number, columns = lst_columns)
df_num.info()
df_num.head()
```
![image](https://github.com/MeenWhile/recognize-handwritten-Thai-numbers/assets/125643589/2122c64a-dd3b-4a04-985b-a226ff70cc8b)
