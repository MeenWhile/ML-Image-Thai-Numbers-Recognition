# recognize-handwritten-Thai-numbers

## Objective

## 1. Data Collection & Import Python Packages

### 1.1 Data Collection
เริ่มต้น เราได้ทำการเก็บรวบรวมไฟล์รูปภาพที่มีการเขียนตัวเลขภาษาไทยตั้งแต่เลข 0 ถึงเลข 9ไว้ภายในรูป 
โดยมีตัวเลขละ 40 รูป รวมแล้วเราจะได้ไฟล์รูปภาพที่มีการเขียนตัวเลขภาษาไทยทั้งหมด 400 รูป

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
