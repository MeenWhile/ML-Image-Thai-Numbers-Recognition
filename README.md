# Recognize handwritten Thai numbers

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

### 2.2 Generate dataFrame

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

## 3. Training Model

เมื่อทำการ cleansing data เสร็จแล้ว ทีนี้เราก็จะมาเริ่มสร้าง Machine Learning Model กัน โดยเริ่มจากการแบ่ง dataFrame ออกเป็น 2 ส่วนนั่นคือ 1.data for modeling(train set) และ 2.unseen data for prediction(test set)

```python
X = df_num.iloc[:,1:]
y = df_num['y']
trainX, testX, trainy, testy = train_test_split(X, y, train_size= 80/100, random_state=42)
print('Data for Modeling: ' + str(trainX.shape[0]))
print('Unseen Data For Predictions: ' + str(testX.shape[0]))
```
![image](https://github.com/MeenWhile/recognize-handwritten-Thai-numbers/assets/125643589/b4459007-e02e-4b0a-a0f3-d36afb43b14e)

จากนั้นก็ check จำนวนของตัวเลขแต่ละตัวใน train set ว่ามีเท่าไหร่บ้าง

```python
y_count = pd.DataFrame(trainy.value_counts())
y_count.reset_index()
```
![image](https://github.com/MeenWhile/recognize-handwritten-Thai-numbers/assets/125643589/ecbdd036-a51f-439f-8d29-3f644e36025b)

เมื่อเตรียมการเสร็จแล้ว เราก็เริ่มสร้าง Machine Learning Model โดยเราเลือกใช้ library pycaret ในการช่วยสร้าง model เนื่องจาก pycaret นั้น สามารถใช้ model ได้หลากหลายแบบ และสามารถคัดเลือก model ที่เหมาะสมที่สุดสำหรับ data ชุดนี้มาได้

```python
exp_clf101 = setup(data = trainX, target = trainy)
```
![image](https://github.com/MeenWhile/recognize-handwritten-Thai-numbers/assets/125643589/deec552c-de99-468d-a5cd-30981f644226)

```python
best = compare_models() #include = ['lr','et','rf','svm','lightgbm','knn','lda','nb','dt','ridge']
```
![image](https://github.com/MeenWhile/recognize-handwritten-Thai-numbers/assets/125643589/abc3d6ae-2a47-4ec2-85ae-328883b35323)

และเมื่อรันได้ถึงตรงนี้ เราก็ได้รู้ว่า model ที่เหมาะสมในการเลือกใช้สำหรับ data ชุดนี้คือ Extra Trees Classifier ซึ่งมีค่า Accuracy อยู่ที่ 0.9733 และมีค่า AUC อยู่ที่ 0.9993 

เมื่อเรารู้แล้วว่า model ที่เหมาะสมที่สุดคือ Extra Trees Classifier เราจึงเริ่มต้นนำ model นี้ไปใช้สร้าง model

```python
print(best)
```
![image](https://github.com/MeenWhile/recognize-handwritten-Thai-numbers/assets/125643589/8fd45a09-80d0-476c-9862-03c102caaa17)

```python
model = create_model(best)
```
![image](https://github.com/MeenWhile/recognize-handwritten-Thai-numbers/assets/125643589/f9990526-6b8b-4a27-b500-07b06ac4aadb)

## 4. Evaluation and Testing Model

### 4.1 Evaluate Model
เมื่อสร้าง model เสร็จ ต่อไปเราก็นำ model ที่ได้มา evaluate

```python
y_scores = model.predict_proba(X)
y_onehot = pd.get_dummies(y, columns=model.classes_)

fig_ROC = go.Figure()
fig_ROC.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

for i in range(y_scores.shape[1]):
    y_true = y_onehot.iloc[:, i]
    y_score = y_scores[:, i]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)

    name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
    fig_ROC.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

fig_ROC.update_layout(
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    yaxis=dict(scaleanchor="x", scaleratio=1),
    xaxis=dict(constrain='domain'),
    width=700, height=500
)
fig_ROC.show()
```
![image](https://github.com/MeenWhile/recognize-handwritten-Thai-numbers/assets/125643589/9f38999a-2236-45aa-97f9-43ad442a3cf7)

```python
actual_labels = np.array(y)
predicted_labels = model.predict(X)

z = confusion_matrix(actual_labels, predicted_labels)

# change each element of z to type string for annotations
z_text = [[str(y) for y in x] for x in z]

# set up figure 
label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
fig_cfm = ff.create_annotated_heatmap(z, x=label, y=label, annotation_text=z_text, colorscale='Blues')

# add title
fig_cfm.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                  #xaxis = dict(title='x'),
                  #yaxis = dict(title='x')
                 )

# add custom xaxis title
fig_cfm.add_annotation(dict(font=dict(color="black",size=14),
                        x=0.5,
                        y=-0.1,
                        showarrow=False,
                        text="Predicted value",
                        xref="paper",
                        yref="paper"))

# add custom yaxis title
fig_cfm.add_annotation(dict(font=dict(color="black",size=14),
                        x=-0.1,
                        y=0.5,
                        showarrow=False,
                        text="Real value",
                        textangle=-90,
                        xref="paper",
                        yref="paper"))

# adjust margins to make room for yaxis title
fig_cfm.update_layout(margin=dict(t=50, l=200))

# add colorbar
fig_cfm['data'][0]['showscale'] = True
fig_cfm.show()
```
![newplot (1)](https://github.com/MeenWhile/recognize-handwritten-Thai-numbers/assets/125643589/0440c579-d54a-4f7d-8a53-567bd15eac05)

```python
Train = pd.DataFrame(trainX, columns = lst_columns)
Train['y'] = trainy
pred_seen = predict_model(model, data = Train)
```
![image](https://github.com/MeenWhile/recognize-handwritten-Thai-numbers/assets/125643589/1b463659-24d2-42e1-b0f7-6e699836c2fa)

โดยผลลัพธ์ที่ได้นั้น เราได้ค่า accuracy เท่ากับ 0.9906 และได้กราฟ ROC กับ กราฟ Confusion matrix ที่ค่อนข้างน่าพอใจ

### 4.2 Testing Model

จากนั้น เราก็นำ model ไป predict กับ test data ต่อเพื่อตรวจสอบว่า model สามารถ predict data ที่ยังไม่เคยเห็นได้ถูกต้องหรือไม่

```python
y_scores_test = model.predict_proba(testX)
y_onehot_test = pd.get_dummies(testy, columns=model.classes_)

fig_ROC_test = go.Figure()
fig_ROC_test.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)
    
for i in range(y_scores.shape[1]):
    y_true_test = y_onehot_test.iloc[:, i]
    y_score_test = y_scores_test[:, i]
        
    fpr_test, tpr_test, _ = roc_curve(y_true_test, y_score_test)
    auc_score_test = roc_auc_score(y_true_test, y_score_test)
        
    name_test = f"{y_onehot.columns[i]} (AUC={auc_score_test:.2f})"
    fig_ROC_test.add_trace(go.Scatter(x=fpr_test, y=tpr_test, name=name_test, mode='lines'))
        
fig_ROC_test.update_layout(
    title_text='<i><b>ROC Curve</b></i>',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    yaxis=dict(scaleanchor="x", scaleratio=1),
    xaxis=dict(constrain='domain'),
    width=700, height=500
)

fig_ROC_test.show()
```
![image](https://github.com/MeenWhile/recognize-handwritten-Thai-numbers/assets/125643589/9c067b6a-7f69-4900-8f4d-d4641cccc6d7)

```python
actual_labels_test = np.array(testy)
predicted_labels_test = model.predict(testX)
        
z_test = confusion_matrix(actual_labels_test, predicted_labels_test)
        
# change each element of z to type string for annotations
z_text_test = [[str(y) for y in x] for x in z_test]
        
# set up figure 
label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
fig_cfm_test = ff.create_annotated_heatmap(z_test, x=label, y=label, annotation_text=z_text_test, colorscale='Blues')
        
# add title
fig_cfm_test.update_layout(title_text='<i><b>Confusion matrix</b></i>\n',
                    #xaxis = dict(title='x'),
                    #yaxis = dict(title='x')
                    )
        
# add custom xaxis title
fig_cfm_test.add_annotation(dict(font=dict(color="black",size=14),
                        x=0.5,
                        y=-0.1,
                        showarrow=False,
                        text="Predicted value",
                        xref="paper",
                        yref="paper"))
        
# add custom yaxis title
fig_cfm_test.add_annotation(dict(font=dict(color="black",size=14),
                        x=-0.1,
                        y=0.5,
                        showarrow=False,
                        text="Real value",
                        textangle=-90,
                        xref="paper",
                        yref="paper"))
        
# adjust margins to make room for yaxis title
fig_cfm_test.update_layout(margin=dict(t=50, l=200))
        
# add colorbar
fig_cfm_test['data'][0]['showscale'] = True

fig_cfm_test.show()
```
![newplot (2)](https://github.com/MeenWhile/recognize-handwritten-Thai-numbers/assets/125643589/f3ebf5ee-89d1-4388-8cb6-a191f2d9dab1)

```python
Test = pd.DataFrame(testX, columns = lst_columns)
Test['y'] = testy
pred_unseen = predict_model(model, data = Test)
```
![image](https://github.com/MeenWhile/recognize-handwritten-Thai-numbers/assets/125643589/36ac6060-0873-4444-a98e-da002def5b9e)

โดยผลลัพธ์ที่ได้สำหรับ test data เราได้ค่า accuracy เท่ากับ 0.9625 ซึ่งมีค่าที่ต่ำกว่า train data เล็กน้อย ถึงอย่างนั้น model ก็สามารถ predict ค่าที่ถูกต้องได้ถึง 77 รูป จากทั้งหมด 80 รูป เราจึงมองว่า model ชุดนี้สามารถนำไปใช้งานได้

## 5. Finalize Model

และสุดท้าย เมื่อเรารู้ได้ว่า model นี้สามารถใช้ได้ เราจึงสร้าง model นี้อีกรอบด้วยกระบวนการเดิม แต่รอบนี้ได้นำรูปภาพทั้งหมด 400 รูปมา train model และ save model ออกมาเพื่อใช้กับรูปภาพใหม่ในอนาคตที่อาจจะได้เจอ

```python
final_model = finalize_model(model)
final_model
```
![image](https://github.com/MeenWhile/recognize-handwritten-Thai-numbers/assets/125643589/5ed5881a-0112-4b42-aa08-c83f72b478c5)

```python
save_model(model, 'thainumber_ml')
```
![image](https://github.com/MeenWhile/recognize-handwritten-Thai-numbers/assets/125643589/58174f91-495b-4557-a2c1-8a513bd605ca)
