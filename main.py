import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# --- STEP 1: CREATE DUMMY DATA (The Learning Material) ---
# We teach the AI by giving it examples.
data = {
    'text': [
        'India wins cricket match against Australia', 'Messi scores a fantastic goal', 'IPL auction 2026 details', # Sports
        'Prime Minister announces new digital policy', 'Election results declared today', 'Parliament session updates', # Politics
        'New iPhone 16 features leaked', 'Artificial Intelligence is replacing coding jobs', 'Python 4.0 release date', # Tech
        'Stock market crashes globally', 'Gold prices reach all time high', 'GDP growth rate increases', # Finance
        'Movie releases this Friday', 'Actor wins best performance award', 'Netflix announces new series' # Entertainment
    ],
    'category': ['Sports', 'Sports', 'Sports', 'Politics', 'Politics', 'Politics', 'Tech', 'Tech', 'Tech', 
                 'Finance', 'Finance', 'Finance', 'Entertainment', 'Entertainment', 'Entertainment']
}

# Convert to a Table
df = pd.DataFrame(data)

# --- STEP 2: TRAIN THE MODEL ---
print("Training the AI Model...")
vectorizer = CountVectorizer() # Converts text to numbers
X = vectorizer.fit_transform(df['text'])

model = MultinomialNB() # The Algorithm (Naive Bayes)
model.fit(X, df['category'])
print("Model Trained Successfully! ✅")

# --- STEP 3: PREDICTION FUNCTION ---
def predict_category(text):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)
    return prediction[0]

# --- STEP 4: RUN THE APP ---
if __name__ == "__main__":
    print("\n--- 📰 NEWS HEADLINE CLASSIFIER ---")
    print("Type a headline to see its category (or type 'quit' to exit)")
    
    while True:
        user_input = input("\n📝 Enter Headline: ")
        if user_input.lower() == 'quit':
            break
        
        category = predict_category(user_input)
        print(f"👉 Predicted Category: {category}")