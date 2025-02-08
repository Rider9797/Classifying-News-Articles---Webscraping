# %%
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk
from sklearn.feature_extraction.text import CountVectorizer
# from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
# from sklearn.preprocessing import LabelEncoder


from sklearn.feature_extraction.text import TfidfVectorizer
# nltk.download('punkt')
# print(nltk.data.path)

# %%
DF  = pd.read_csv('afterEDAFinal.csv')

# %%
from keras.preprocessing.sequence import pad_sequences

X = DF['content']

Y = DF['gold_label']
Y = Y.replace({'entertainment':0,'business':1,'sports':2,'science-technology':3,'world':4})


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


X_train = X_train.astype(str)
X_test = X_test.astype(str)


tokenizer = Tokenizer(nb_words=30000, char_level=False)
tokenizer.fit_on_texts(X_train)

sequences = tokenizer.texts_to_sequences(X_train)
test_sequences = tokenizer.texts_to_sequences(X_test)


X_train = pad_sequences(sequences, maxlen = 300, padding ='post', truncating = 'post')
X_test = pad_sequences(test_sequences, maxlen = 300, padding ='post', truncating = 'post')

X_train = torch.tensor(X_train, dtype=torch.long)
X_test = torch.tensor(X_test,dtype=torch.long)

Y_train = torch.tensor(Y_train.values, dtype=torch.long)
Y_test = torch.tensor(Y_test.values, dtype=torch.long)


unique_labels = torch.unique(Y_train)



print(unique_labels)

# %%
import torch.nn as nn
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Linear(embed_size, 256)
        self.relu = nn.ReLU() 
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim = 1)  
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


vocab_size = 30000
embed_size = 300
num_classes = 5
# print(num_classes)
model = TextClassifier(vocab_size, embed_size, num_classes)

# %%
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)


for epoch in range(100):  
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, Y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")



# %%
# model.eval()

with torch.no_grad():
    
    outputs = model(X_test)
    _, predicted_classes = torch.max(outputs, dim=1)


y_pred = predicted_classes.cpu().numpy()
y_true = Y_test.cpu().numpy()


accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted') 
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print("Test Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred))


