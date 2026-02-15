"""
Quick Start Script
Simple script to test the text classification pipeline
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.text_processor import TextPreprocessor, FeatureExtractor
from src.models.baseline import BaselineModel, ModelComparison
from src.utils.helpers import set_seed, plot_class_distribution

# Set random seed
set_seed(42)

print("=" * 70)
print("TEXT CLASSIFICATION - QUICK START")
print("=" * 70)

# 1. Create sample dataset
print("\n1. Creating sample dataset...")
sample_data = {
    'text': [
        'This product is absolutely amazing! Best purchase ever.',
        'Terrible experience. Would not recommend to anyone.',
        'Pretty good overall, meets my expectations.',
        'Worst service I have ever received. Very disappointed.',
        'Excellent quality! Highly satisfied with this product.',
        'Not worth the money. Poor quality materials.',
        'Fast delivery and great customer service!',
        'Disappointed with the performance. Expected better.',
        'Love it! Will definitely buy again.',
        'Complete waste of money. Do not buy.'
    ] * 10,  # 100 samples
    'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 10
}

df = pd.DataFrame(sample_data)
print(f"Dataset size: {len(df)} samples")
print(f"Classes: {df['label'].unique()}")
print(f"Class distribution: {df['label'].value_counts().to_dict()}")

# 2. Preprocessing
print("\n2. Preprocessing texts...")
preprocessor = TextPreprocessor()
df_processed = preprocessor.preprocess_dataframe(df, 'text', 'label')

print(f"Processed samples: {len(df_processed)}")
print("\nExample:")
print(f"Original: {df['text'].iloc[0]}")
print(f"Cleaned:  {df_processed['cleaned_text'].iloc[0]}")

# 3. Feature extraction
print("\n3. Extracting features...")
extractor = FeatureExtractor(method='tfidf', max_features=100)
X = extractor.fit_transform(df_processed['cleaned_text'])
y = df_processed['label'].values

print(f"Feature matrix shape: {X.shape}")

# 4. Split data
print("\n4. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape[0]} samples")
print(f"Test:  {X_test.shape[0]} samples")

# 5. Train baseline models
print("\n5. Training baseline models...")
print("-" * 70)

comparison = ModelComparison()
comparison.add_model("Logistic Regression", "logistic")
comparison.add_model("Naive Bayes", "naive_bayes")
comparison.add_model("Random Forest", "random_forest", n_estimators=50)

# Train
comparison.train_all(X_train, y_train)

# Evaluate
print("\n6. Evaluating models...")
print("-" * 70)
results = comparison.evaluate_all(X_test, y_test)

# Get best model
best_name, best_model, best_score = comparison.get_best_model('f1_score')

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")

print("\n" + "=" * 70)
print(f"🏆 BEST MODEL: {best_name}")
print(f"   F1-Score: {best_score:.4f}")
print("=" * 70)

# 7. Test prediction
print("\n7. Testing prediction on new samples...")
print("-" * 70)

test_samples = [
    "This is the best product I have ever bought!",
    "Absolutely terrible. Don't waste your money.",
    "It's okay, nothing special."
]

for i, sample in enumerate(test_samples, 1):
    # Preprocess
    cleaned = preprocessor.preprocess(sample)
    # Extract features
    features = extractor.transform([cleaned])
    # Predict
    prediction = best_model.predict(features)[0]
    
    print(f"\nSample {i}: {sample}")
    print(f"Prediction: {'Positive' if prediction == 1 else 'Negative'}")
    
    # Get probability if available
    try:
        proba = best_model.predict_proba(features)[0]
        print(f"Confidence: {proba[prediction]:.2%}")
    except:
        pass

print("\n" + "=" * 70)
print("✅ QUICK START COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\nNext steps:")
print("1. Load your own dataset")
print("2. Perform detailed EDA using notebooks/01_EDA.ipynb")
print("3. Train deep learning models using main.py")
print("4. Run demo using: streamlit run demo/app.py")
print("=" * 70)
