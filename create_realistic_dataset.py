import pandas as pd
import numpy as np
import random

# Set random seed
np.random.seed(42)
random.seed(42)

# More realistic product review templates with variations
positive_templates = [
    # Short positive
    "Great product!",
    "Love it!",
    "Amazing!",
    "Perfect!",
    "Excellent!",
    
    # Medium positive
    "This product exceeded my expectations. Very happy with the purchase.",
    "Fantastic quality and fast delivery. Highly recommend!",
    "Best purchase I've made this year. Worth every penny.",
    "Excellent product with great features. Very satisfied customer.",
    "Outstanding quality! This is exactly what I was looking for.",
    "Very impressed with the build quality and performance.",
    "Great value for money. Works perfectly as described.",
    "Superb! This product has made my life so much easier.",
    "Highly recommend this to anyone looking for quality.",
    "Perfect for my needs. Couldn't ask for more.",
    
    # Long positive
    "I bought this product last month and I'm extremely satisfied. The quality is top-notch and it works exactly as advertised. Shipping was fast and the packaging was secure. Would definitely buy from this seller again!",
    "This is hands down the best product in its category. I've tried several similar products before, but this one stands out. The attention to detail is impressive and the customer service was excellent. Highly recommended!",
    "After using this product for a few weeks, I can confidently say it's worth the investment. The build quality is solid, it's easy to use, and it does exactly what it promises. Very happy customer here!",
    "Great product at an affordable price! I was skeptical at first, but after reading the reviews I decided to give it a try. I'm so glad I did! It works wonderfully and has exceeded my expectations.",
    "Absolutely love this product! The quality is amazing and it arrived much faster than expected. The seller was very responsive to my questions. Overall, a fantastic buying experience!",
]

negative_templates = [
    # Short negative
    "Poor quality.",
    "Don't buy!",
    "Waste of money.",
    "Disappointed.",
    "Terrible!",
    
    # Medium negative
    "Very disappointed with this purchase. Not as described.",
    "Poor quality materials. Broke after just a few uses.",
    "Not worth the price. Expected much better quality.",
    "Terrible product. Does not work as advertised at all.",
    "Very low quality. Looks and feels cheap.",
    "Awful experience. Product arrived damaged and defective.",
    "Don't waste your money. Complete disappointment.",
    "Poor build quality. Not what I expected for the price.",
    "Very unsatisfied. Would not recommend to anyone.",
    "Horrible quality. Returned it immediately.",
    
    # Long negative  
    "I ordered this product with high hopes based on the reviews, but I was severely disappointed. The quality is much worse than advertised. It broke within the first week of normal use. Customer service was unhelpful when I tried to return it. Save your money and look elsewhere!",
    "This is by far the worst purchase I've made online. The product looks nothing like the pictures. The materials are cheap and flimsy. It stopped working after just two days. The seller refused to provide a refund. Absolutely terrible experience!",
    "Very poor quality control. The product arrived with several defects and didn't work properly from day one. I tried contacting customer service multiple times but received no response. Would give zero stars if I could. Stay away from this product!",
    "Complete waste of money. I should have read the negative reviews more carefully. The product is poorly made and doesn't function as described. Shipping took forever and the packaging was inadequate. Very disappointed and will not buy from this seller again.",
    "Extremely dissatisfied with this purchase. The quality is subpar and it feels like a cheap knockoff. It didn't meet any of my expectations. The description was misleading and the photos don't represent the actual product. Would not recommend!",
]

# Additional variations for more natural text
positive_additions = [
    " Great purchase!",
    " Will buy again.",
    " Highly recommended!",
    " Five stars!",
    " Love it!",
    " Very happy!",
    " Excellent!",
    " Amazing quality!",
    " Perfect!",
    " Best decision!",
]

negative_additions = [
    " Very disappointed.",
    " Not recommended.",
    " Waste of money.",
    " Poor quality.",
    " Don't buy.",
    " Terrible experience.",
    " Regret buying.",
    " Not worth it.",
    " Awful.",
    " Stay away!",
]

# Generate dataset
reviews = []
labels = []

# Generate 1000 positive reviews
for i in range(1000):
    # Select random template
    review = random.choice(positive_templates)
    
    # 30% chance to add additional phrase
    if random.random() < 0.3:
        review += random.choice(positive_additions)
    
    reviews.append(review)
    labels.append(1)

# Generate 1000 negative reviews
for i in range(1000):
    # Select random template
    review = random.choice(negative_templates)
    
    # 30% chance to add additional phrase
    if random.random() < 0.3:
        review += random.choice(negative_additions)
    
    reviews.append(review)
    labels.append(0)

# Create DataFrame
df = pd.DataFrame({
    'text': reviews,
    'label': labels
})

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Add a unique ID
df.insert(0, 'id', range(1, len(df) + 1))

# Save to CSV
output_path = 'data/raw/product_reviews_2000.csv'
import os
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print("=" * 70)
print("DATASET CREATED SUCCESSFULLY!")
print("=" * 70)
print(f"\n📊 Dataset Statistics:")
print(f"   Total samples: {len(df)}")
print(f"   Positive samples (label=1): {(df['label'] == 1).sum()}")
print(f"   Negative samples (label=0): {(df['label'] == 0).sum()}")
print(f"   Class balance: {(df['label'] == 1).sum() / len(df) * 100:.1f}% positive")

print(f"\n📝 Text Statistics:")
text_lengths = df['text'].str.len()
word_counts = df['text'].str.split().str.len()
print(f"   Average text length: {text_lengths.mean():.1f} characters")
print(f"   Average word count: {word_counts.mean():.1f} words")
print(f"   Min words: {word_counts.min()}")
print(f"   Max words: {word_counts.max()}")

print(f"\n💾 File saved to: {output_path}")

print(f"\n🔍 Sample reviews:")
print("-" * 70)
for idx in [0, 5, 10, 15, 20]:
    row = df.iloc[idx]
    label_name = "Positive" if row['label'] == 1 else "Negative"
    print(f"\n[{label_name}] {row['text'][:100]}...")

print("\n" + "=" * 70)
print("✅ Ready to use! Run: python main.py --data data/raw/product_reviews_2000.csv")
print("=" * 70)
