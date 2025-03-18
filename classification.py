import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from urllib.request import urlopen
import optuna  # Import Optuna for hyperparameter optimization

# Download UCI SMS Spam Collection dataset
print("Downloading SMS Spam Collection dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
try:
    import io
    import zipfile
    
    response = urlopen(url)
    zipdata = zipfile.ZipFile(io.BytesIO(response.read()))
    data_file = zipdata.open('SMSSpamCollection')
    
    # Read the data
    data = pd.read_csv(data_file, sep='\t', header=None, names=['label', 'text'])
    print("Dataset downloaded and loaded successfully.")
except Exception as e:
    print(f"Failed to download dataset: {e}")
    print("Attempting to use a hardcoded dataset sample...")
    
    # Fallback: Use a sample of the dataset
    data = pd.DataFrame([
        ['ham', 'Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...'],
        ['ham', 'Ok lar... Joking wif u oni...'],
        ['spam', 'Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C'],
        ['ham', 'U dun say so early hor... U c already then say...'],
        ['ham', 'Nah I don\'t think he goes to usf, he lives around here though'],
        ['spam', 'FreeMsg Hey there darling it\'s been 3 week\'s now and no word back! I\'d like some fun you up for it still? Tb ok! XxX std chgs to send'],
        ['ham', 'Even my brother is not like to speak with me. They treat me like aids patent.'],
        ['ham', 'As per your request "Melle Melle (Oru Minnaminunginte Nurungu Vettam)" has been set as your callertune for all Callers.'],
        ['spam', 'WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.'],
        ['spam', 'Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for FREE! Call The Mobile Update Co FREE on 08002986030'],
        ['ham', 'I\'m gonna be home soon and i don\'t want to talk about this stuff anymore tonight, k? I\'ve cried enough today.'],
        ['spam', 'SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply'],
        ['spam', 'URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C'],
        ['ham', 'I\'ve been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise.'],
        ['ham', 'I HAVE A DATE ON SUNDAY WITH WILL!!'],
        ['spam', 'XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap.xxxmobile.com/n/?3qkj'],
        ['ham', 'Oh k...i\'m watching here:)'],
        ['ham', 'Eh u remember how 2 spell his name... Yes i did. He v naughty make until i v wet.']
    ], columns=['label', 'text'])
    print("Using sample dataset instead.")

# Display dataset info
print("Dataset shape:", data.shape)
print("First few rows:")
print(data.head())

# Check class distribution
print("\nClass distribution:")
print(data['label'].value_counts())

# Preprocess text function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()

# Clean texts
print("\nPreprocessing text...")
data['processed_text'] = data['text'].apply(preprocess_text)

# Features and labels
X = data['processed_text']
y = data['label']

# Visualize text length distribution
plt.figure(figsize=(10, 5))
data['text_length'] = data['processed_text'].apply(len)
sns.histplot(data=data, x='text_length', hue='label', bins=50, kde=True)
plt.title('Distribution of Text Lengths by Class')
plt.xlabel('Text Length')
plt.xlim(0, 200)  # Focus on the majority of the data
plt.savefig('text_length_distribution.png')
plt.close()

# Most common words in spam vs ham
def get_top_words(texts, n=20):
    words = ' '.join(texts).split()
    word_counts = {}
    for word in words:
        if len(word) > 3:  # Ignore short words
            word_counts[word] = word_counts.get(word, 0) + 1
    return sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:n]

spam_words = get_top_words(data[data['label'] == 'spam']['processed_text'])
ham_words = get_top_words(data[data['label'] == 'ham']['processed_text'])

print("\nTop words in spam messages:")
for word, count in spam_words:
    print(f"{word}: {count}")

print("\nTop words in ham messages:")
for word, count in ham_words:
    print(f"{word}: {count}")

# TF-IDF vectorization
print("\nVectorizing text...")
vectorizer = TfidfVectorizer(
    stop_words='english',
    min_df=5,  # Ignore terms that appear in less than 5 documents
    max_df=0.7,  # Ignore terms that appear in more than 70% of documents
    max_features=5000  # Limit to top 5000 features
)
X_tfidf = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# Define the objective function for Optuna
def objective(trial):
    # Define hyperparameters to optimize
    alpha = trial.suggest_float('alpha', 0.001, 10.0, log=True)
    
    # Create and train the model
    nb_model = MultinomialNB(alpha=alpha)
    
    # Use cross-validation for more robust evaluation
    scores = cross_val_score(nb_model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    
    # Return the mean accuracy
    return scores.mean()

# Create and run the Optuna study
print("\nStarting hyperparameter optimization with Optuna...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)  # Adjust n_trials as needed

# Print optimization results
print("\nOptimization Results:")
print(f"Best alpha parameter: {study.best_params['alpha']}")
print(f"Best accuracy: {study.best_value:.4f}")

# Train final model with best parameters
final_model = MultinomialNB(alpha=study.best_params['alpha'])
final_model.fit(X_train, y_train)

# Make predictions
y_pred = final_model.predict(X_test)

# Evaluate model
print("\nModel evaluation:")
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix visualization
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Visualization of hyperparameter importance
plt.figure(figsize=(10, 6))
optuna.visualization.matplotlib.plot_param_importances(study)
plt.title('Hyperparameter Importance')
plt.tight_layout()
plt.savefig('param_importances.png')
plt.close()

# Visualization of optimization history
plt.figure(figsize=(10, 6))
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.title('Optimization History')
plt.tight_layout()
plt.savefig('optimization_history.png')
plt.close()

# Example predictions
def predict_message(message, vectorizer, model):
    processed = preprocess_text(message)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0]
    return prediction, max(proba)

# Test with example messages
example_messages = [
    "Congratulations! You've won a $1000 gift card. Call now to claim your prize!",
    "Hey, are we still meeting for coffee at 3pm today?",
    "URGENT: Your account has been suspended. Click here to reactivate",
    "I'll be home in 20 minutes, need anything from the store?"
]

print("\nExample predictions:")
for message in example_messages:
    prediction, confidence = predict_message(message, vectorizer, final_model)
    print(f"Message: {message}")
    print(f"Prediction: {prediction} (Confidence: {confidence:.2f})")
    print("-" * 50)