<H3>BALA MURUGAN S</H3> 
<H3>212223230027</H3> 
<H3>EX. NO.1</H3>
<H3>28-10-2026</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```
```python
df = pd.read_csv("/content/Churn_Modelling.csv")
df
```
```python
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
df
```
```python
df = df.dropna()
```
```python
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df = pd.get_dummies(df, columns=["Geography"], drop_first=True, dtype=int)
```
```python
x = df.drop("Exited", axis=1)
y = df["Exited"]
```
```python
xt, xst, yt, yst = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
```
```python
scaler = StandardScaler()
xt = scaler.fit_transform(xt)
xst = scaler.transform(xst)
```
```python
print("Training Data:", xt.shape)
print("Testing Data:", xst.shape)
print("Training Labels:", yt.shape)
print("Testing Labels:", yst.shape)
```


## OUTPUT:
<img width="1620" height="598" alt="image" src="https://github.com/user-attachments/assets/03435884-32a2-4e8a-97dd-807b181229bb" />

<img width="1317" height="507" alt="image" src="https://github.com/user-attachments/assets/ca4c7655-4a9e-4c0b-9e99-b96b1362af35" />

<img width="567" height="215" alt="image" src="https://github.com/user-attachments/assets/f0c435f8-4ec2-43a2-84a1-69d8423a5294" />




## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


