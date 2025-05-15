import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('bank.csv')

X = df.iloc[:, :-1]  # All columns except the last one (features)
y = df.iloc[:, -1]   # Last column (species)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Function to get user input and predict the class
def get_prediction():
    print("Enter the following details about the flower:")
    age = float(input("Age (years): "))
    income = float(input("Income ($): "))
    loan_amount = float(input("Loan Amount ($): "))
    
    
    # Create the sample with user input
    sample = pd.DataFrame([[age, income, loan_amount]], columns=X.columns)
    
    # Predict the class
    predicted_class = model.predict(sample)[0]
    print(f"Prediction if you will get loan: {predicted_class}")

# Call the function to get prediction from the user
get_prediction()