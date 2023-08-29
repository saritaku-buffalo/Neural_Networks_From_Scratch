#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Assignment_2 Part-1


# In[1]:


import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv("dataset.csv")


# In[ ]:


df


# In[ ]:


#remove rows with string characters from dataframe
df = df[np.isfinite(pd.to_numeric(df.f1, errors="coerce"))]
df = df[np.isfinite(pd.to_numeric(df.f2, errors="coerce"))]
df = df[np.isfinite(pd.to_numeric(df.f3, errors="coerce"))]
df = df[np.isfinite(pd.to_numeric(df.f4, errors="coerce"))]
df = df[np.isfinite(pd.to_numeric(df.f5, errors="coerce"))]
df = df[np.isfinite(pd.to_numeric(df.f6, errors="coerce"))]
df = df[np.isfinite(pd.to_numeric(df.f7, errors="coerce"))]


# In[ ]:


df


# In[9]:


df['f1'] = df['f1'].astype(str).astype(float)
df['f2'] = df['f2'].astype(str).astype(float)
df['f3'] = df['f3'].astype(str).astype(float)
df['f4'] = df['f4'].astype(str).astype(float)
df['f5'] = df['f5'].astype(str).astype(float)
df['f6'] = df['f6'].astype(str).astype(float)
df['f7'] = df['f7'].astype(str).astype(float)


# In[10]:


df


# In[11]:


numerical_columns = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])


# In[12]:


df = pd.DataFrame(df)


# In[13]:


df


# In[15]:


df.describe()


# In[16]:


df_train, df_test = train_test_split(df, test_size=0.2, random_state = 3)


# In[17]:


df_train


# In[18]:


df_test


# In[21]:


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(7, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.fc4 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

input_size = len(df_train.columns) - 1 # number of features

output_size = 1 # binary classification

model = NeuralNetwork()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 50
batch_size = 64

for epoch in range(epochs):
    running_loss = 0.0
    j = 0
    for i in range(0, len(df_train), batch_size):
        # get the inputs and labels for this batch
        inputs = torch.Tensor(df_train.iloc[i:i+batch_size, :-1].values)
        labels = torch.Tensor(df_train.iloc[i:i+batch_size, -1].values)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels.view(-1, 1))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        j += 1

    print(f"Epoch {epoch+1}, loss: {running_loss / j}")

# evaluate the model on testing data
test_inputs = torch.Tensor(df_test.iloc[:, :-1].values)
test_labels = torch.Tensor(df_test.iloc[:, -1].values)
test_outputs = model(test_inputs)
train_inputs = torch.Tensor(df_train.iloc[:, :-1].values)
train_labels = torch.Tensor(df_train.iloc[:, -1].values)
train_outputs = model(train_inputs)
# test_loss = criterion(test_outputs, test_labels.view(-1, 1))
# print(test_loss)

final_arr1 = []
for i in test_outputs.detach().numpy():
    if i > 0.5:
        final_arr1.append(1)
    else:
        final_arr1.append(0)
final_arr2 = []
for i in train_outputs.detach().numpy():
    if i > 0.5:
        final_arr2.append(1)
    else:
        final_arr2.append(0)
    
# print(final_arr)
    
test_acc = np.mean(final_arr1 == test_labels.detach().numpy())
print(f"Test accuracy: {test_acc}")
train_acc = np.mean(final_arr2 == train_labels.detach().numpy())
print(f"Train accuracy: {train_acc}")


# In[22]:


# 3 Visualization
plt.figure(figsize=(8, 6))
plt.hist(test_inputs[:, 1], bins=20)
plt.xlabel('Test Inputs')
plt.show()





# In[ ]:


plt.boxplot(df.values)
plt.xticks(range(1, len(df.columns) + 1), df.columns)
plt.ylabel('Value')
plt.title('Box plot')
plt.show()


# In[ ]:


from sklearn.metrics import confusion_matrix

# convert model outputs to binary predictions
predictions = (test_outputs > 0.5).float().detach().numpy()

# create confusion matrix
cm = confusion_matrix(test_labels, predictions)

# visualize confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.4)
sns.heatmap(cm, annot=True, annot_kws={"size": 16}, cmap='Blues', fmt='g')

plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()







