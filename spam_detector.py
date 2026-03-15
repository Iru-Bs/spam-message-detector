import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from pathlib import Path

# Load dataset
# Prefer a dataset file in the same directory as this script.
base_dir = Path(__file__).resolve().parent
csv_path = base_dir / "spam.csv"
ts_path = base_dir / "sms.tsv"

if csv_path.exists():
    data_path = csv_path
elif ts_path.exists():
    data_path = ts_path
else:
    raise FileNotFoundError(
        f"Could not find dataset file. Expected one of: {csv_path}, {ts_path}"
    )

# The dataset is tab-separated and has two columns: label and message.
data = pd.read_csv(data_path, sep="\t", names=["label", "message"])

# Convert labels
data["label"] = data["label"].map({"ham":0, "spam":1})

X = data["message"]
y = data["label"]

# Convert text to numbers
vectorizer = CountVectorizer()
X_vector = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vector, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", round(accuracy,2))

# User input
msg = input("Enter message: ")

msg_vector = vectorizer.transform([msg])

prediction = model.predict(msg_vector)

if prediction[0] == 1:
    print("Result: Spam Message")
else:
    print("Result: Normal Message")