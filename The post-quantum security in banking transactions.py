# -*- coding: utf-8 -*-
"""
Created on Thu May  9 18:09:26 2024

@author: Maha
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler , StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report , confusion_matrix
from sklearn.metrics import accuracy_score , f1_score
from pqcrypto.kem.kyber512 import generate_keypair, encapsulate, decapsulate
from oqs import KEM
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import pickle
import sys
import numpy as np
from oqs import KEM
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad



 # Read the Dataset using Pnadas
dataset = pd.read_csv("./creditcard_2023.csv").set_index("id")

# Dsiplay the Data from the Dataset
dataset.sample(2)
 # Display the Shape of the Dataset
print("Here is the Shape of the Datset : {}".format(dataset.shape))
 # Display the Information about the Dataset
print("Here is the Information About Dataset : ")
dataset.info()
# Display the Summary of the Numerical Columns
print("Here is the Summary of the Numerical Columns Data : ")
dataset.describe()
 # Find out the Missing Values from the Dataset
missingValues = dataset.isna().sum()

# Display the Missing Values for Each Columns
print(f"Here is the Missing Values of Each Column :\n{missingValues}")
# Find out the Duplicated Rows from the Dataset
duplicatedData = dataset.duplicated().sum()

# Display the total count of Duplicated Rows
print("Here is the Total Count of Duplicated Row from the Dataset : {}".format(duplicatedData))
 # Drop the Duplicate Data from the Dataset
dataset.drop_duplicates(inplace = True)

# After Drop the Duplicated Data again find out the Duplicated Rows
duplicatedData = dataset.duplicated().sum()

# Display the total count of Duplicated Rows
print("Here is the AgainTotal Count of Duplicated Row from the Dataset : {}".format(duplicatedData))
 # Convert the Dataset into Indepedent and Dependent
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Display the Shape of the X & y Matrix
print("Here is the Shape of X : {}".format(X.shape))
print("Here is the Shape of Y : {}".format(y.shape))
 # Normalize the Dataset Data (Convert Each Column Data into Same Range)

# Create the Object of StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Split the Dataset into Training and Testing
xTrain , xTest , yTrain , yTest = train_test_split(X , y , test_size = 0.25 , random_state = 42)

# Display the Shape of the Training and Testing Data
print("Here is the Shape of X Train : {}".format(xTrain.shape))
print("Here is the Shape of Y Train : {}".format(yTrain.shape))
print("Here is the Shape of X Test  : {}".format(xTest.shape))
print("Here is the Shape of Y Test  : {}".format(yTest.shape))
 # Create the Object for OneClassSVM
svmModel = OneClassSVM(nu = 0.95 , kernel = "linear" , gamma=0.001)

# Now fit the Model in the Training Dataset
svmModel.fit(xTrain , yTrain)
# Predict the Results
prediction = svmModel.predict(xTest)

# Replace the -1 to 0
prediction = np.where(prediction == -1, 0, prediction)

# Calculate the Accuracy of the Model
accModel = accuracy_score(yTest , prediction)

# Display the Accuracy of the Model
print(f"Here is the Accuracy of the Model : {accModel}")
# Calculate the Confusion Matrix for the Model
sns.heatmap(confusion_matrix(yTest , prediction) , annot = True , cbar = True , fmt = "d")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()
# Create the Object for Decision Tree Classifier
Decision = DecisionTreeClassifier()

# Now Fit the Model in the Training Dataset
Decision.fit(xTrain , yTrain)
 # Predict the Results
prediction = Decision.predict(xTest)

# Calculate the Accuracy of the Model
accModel = accuracy_score(yTest , prediction)

# Display the Accuracy of the Model
print(f"Here is the Accuracy of the Model : {accModel}")
 # Calculate the Confusion Matrix for the Model
sns.heatmap(confusion_matrix(yTest , prediction) , annot = True , cbar = True , fmt = "d")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()
# Save the Model with the Pickle Module

# SVM Model
with open("svm_model.pkl", "wb") as f:
  pickle.dump(svmModel , f)

# Decison Tree Classifier Model
with open("decision_tree_model.pkl" , "wb") as f:
  pickle.dump(Decision , f)
  
# Load the model from the file

# SVM Model
with open("svm_model.pkl", 'rb') as file:
  svm_model = pickle.load(file)

# Decision Tree Model
with open("decision_tree_model.pkl" , "rb") as file:
  decision_tree = pickle.load(file)
  

# Function to generate public and private keys using Kyber512 for secure key exchange
def simulate_pqc_key_exchange():
    kem = KEM('Kyber512')
    public_key, secret_key = kem.keypair()
    return public_key, secret_key

# Function to encrypt data using a symmetric key derived from a public key
def encrypt_data(data, public_key):
    kem = KEM('Kyber512')
    # Encapsulate to generate a ciphertext and a shared secret for encryption
    ciphertext, shared_secret_enc = kem.encap(public_key)
    # Use AES-CBC mode for encryption with the derived symmetric key
    cipher = AES.new(shared_secret_enc, AES.MODE_CBC, get_random_bytes(AES.block_size))
    encrypted_data = cipher.encrypt(pad(data, AES.block_size))
    # Return the IV, encrypted data, and encapsulation ciphertext
    return cipher.iv + encrypted_data, ciphertext

# Function to decrypt data using a symmetric key derived from a secret key and encapsulation ciphertext
def decrypt_data(encrypted_data, secret_key, encapsulated_ciphertext):
    kem = KEM('Kyber512')
    # Decapsulate to recover the shared secret for decryption
    shared_secret_dec = kem.decap(encapsulated_ciphertext, secret_key)
    iv = encrypted_data[:AES.block_size]
    encrypted_content = encrypted_data[AES.block_size:]
    cipher = AES.new(shared_secret_dec, AES.MODE_CBC, iv)
    decrypted_data = unpad(cipher.decrypt(encrypted_content), AES.block_size)
    return decrypted_data

# Generate keys
public_key, private_key = simulate_pqc_key_exchange()

# Simulate a decision-making process 
predict = Decision.predict(xTest[0].reshape(1,-1))[0]

print("Press 1 : For Actual Predict Data : ")
print("Press 2 : For Cipher Text after Encryption : ")
option = int(input("Enter Your Option==>"))

if option == 1:
    if predict == 1:
        print("\nThis Transaction is Fraudulent: {}".format(predict))
    else:
        print("\nThis Transaction is Not Fraudulent: {}".format(predict))
elif option == 2:
    data_bytes = np.array([predict]).astype(np.float32).tobytes()
    encrypted_data, encapsulated_ciphertext = encrypt_data(data_bytes, public_key)
    print("\nHere is the Cipher Text After Encryption: {}".format(encrypted_data.hex()))

    print("\nPress 3 : If you want to Decrypt the Data : ")
    print("Press 4 : To End the Program : ")
    option = int(input("Enter the option==>"))
    if option == 3:
        decrypted_data = decrypt_data(encrypted_data, private_key, encapsulated_ciphertext)
        final_predict = np.frombuffer(decrypted_data, dtype=np.float32)[0]
        print("\nHere is the Final Predict Result: {}".format(final_predict))
    elif option == 4:
        print("Program Ended.")
  
  
  
  
  
  
  