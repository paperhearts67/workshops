import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('iris.csv')

X = df.iloc[:, :-1]  # All columns except the last one (features)
y = df.iloc[:, -1]   # Last column (species)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Function to get user input and predict the class
def get_prediction():
    print("Enter the following details about the flower:")
    sepal_length = float(input("Sepal Length (cm): "))
    sepal_width = float(input("Sepal Width (cm): "))
    petal_length = float(input("Petal Length (cm): "))
    petal_width = float(input("Petal Width (cm): "))
    
    # Create the sample with user input
    sample = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=X.columns)
    
    # Predict the class
    predicted_class = model.predict(sample)[0]
    print(f"Predicted class for the entered flower: {predicted_class}")

# Call the function to get prediction from the user
get_prediction()
