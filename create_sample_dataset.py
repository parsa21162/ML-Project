import pandas as pd
import numpy as np

# Set random seed
np.random.seed(42)

# Positive reviews templates
positive_reviews = [
    "This product is absolutely amazing! Highly recommend it.",
    "Excellent quality and fast shipping. Very satisfied!",
    "Best purchase I've made in a long time. Love it!",
    "Outstanding product! Exceeded my expectations.",
    "Great value for money. Will definitely buy again.",
    "Fantastic! Works perfectly as described.",
    "Superb quality! Very happy with this purchase.",
    "Amazing product! Totally worth the price.",
    "Incredible! This is exactly what I needed.",
    "Perfect! Couldn't be happier with my purchase.",
    "Brilliant product! Five stars all the way.",
    "Wonderful experience! Great customer service too.",
    "Top quality! Very impressed with the results.",
    "Exceptional product! Better than expected.",
    "Love everything about it! Highly recommended.",
    "Outstanding performance! Very satisfied customer.",
    "Absolutely perfect! No complaints whatsoever.",
    "Really happy with this purchase. Great product!",
    "Impressive quality and durability. Love it!",
    "Excellent product! Worth every penny.",
    "Very pleased with the quality and service.",
    "Great product at a reasonable price!",
    "Highly effective and easy to use.",
    "Superb! This has made my life so much easier.",
    "Best quality I've found. Highly recommend!",
    "Amazing value! Can't believe how good this is.",
    "Perfect for my needs. Very satisfied!",
    "Fantastic product! Delivery was quick too.",
    "Love the quality and attention to detail.",
    "Excellent purchase! Would buy again.",
    "This exceeded all my expectations!",
    "Very impressed! Great build quality.",
    "Perfect condition and works wonderfully!",
    "Great product! My family loves it too.",
    "Brilliant! Exactly as advertised.",
    "Outstanding! This is a game changer.",
    "Top-notch quality! Very happy customer.",
    "Excellent! Better than similar products.",
    "Amazing! This solved my problem perfectly.",
    "Fantastic quality for the price!",
    "Very satisfied with this product!",
    "Great features and easy to use!",
    "Love it! Works like a charm.",
    "Superb craftsmanship and design!",
    "Highly recommended! Best decision ever.",
    "Perfect product! No issues at all.",
    "Excellent performance! Very reliable.",
    "Amazing quality! Looks great too.",
    "Very happy! Exactly what I wanted.",
    "Brilliant purchase! Five stars!"
]

# Negative reviews templates
negative_reviews = [
    "Terrible product. Complete waste of money.",
    "Very disappointed with the quality. Would not recommend.",
    "Poor quality and overpriced. Not worth it.",
    "Awful experience. Product broke after one use.",
    "Don't waste your money on this. Total disappointment.",
    "Horrible quality. Nothing like the description.",
    "Very poor craftsmanship. Cheaply made.",
    "Disappointed. Not what I expected at all.",
    "Terrible value. Definitely returning this.",
    "Poor quality materials. Fell apart quickly.",
    "Not worth the price. Very dissatisfied.",
    "Awful product. Does not work as advertised.",
    "Extremely disappointed. Save your money.",
    "Bad quality and poor customer service.",
    "Completely useless. Total waste of time.",
    "Very poor design. Uncomfortable to use.",
    "Terrible. Stopped working after a week.",
    "Horrible experience. Would give zero stars.",
    "Poor performance. Not recommended at all.",
    "Disappointing quality. Feels very cheap.",
    "Not satisfied. Product is defective.",
    "Awful! Doesn't do what it claims.",
    "Very low quality. Returned immediately.",
    "Bad purchase. Regret buying this.",
    "Terrible build quality. Broke easily.",
    "Poor value for money. Not impressed.",
    "Disappointing. Doesn't work properly.",
    "Horrible quality. Looks nothing like pictures.",
    "Very unsatisfied. Product is faulty.",
    "Awful experience from start to finish.",
    "Poor product. Customer service unhelpful.",
    "Terrible quality. Cheaply manufactured.",
    "Very disappointed. Not as described.",
    "Bad design and poor functionality.",
    "Horrible. Broke on first use.",
    "Poor quality control. Defective item.",
    "Terrible purchase. Total disappointment.",
    "Very poor materials. Not durable at all.",
    "Awful. Doesn't meet basic standards.",
    "Disappointing product. Waste of money.",
    "Bad quality. Returned for refund.",
    "Poor craftsmanship. Looks cheap.",
    "Terrible experience. Would not buy again.",
    "Very dissatisfied. Product is useless.",
    "Horrible quality for the price.",
    "Poor performance. Does not work well.",
    "Disappointing purchase. Not recommended.",
    "Awful product. Complete failure.",
    "Very poor quality. Broke immediately.",
    "Bad investment. Regret this purchase."
]

# Generate dataset
reviews = []
labels = []

# Add 500 positive reviews with variations
for i in range(500):
    base_review = np.random.choice(positive_reviews)
    # Add some natural variation
    if np.random.random() > 0.7:
        base_review = base_review + " " + np.random.choice([
            "Great product!", "Love it!", "Highly recommended!",
            "Very satisfied!", "Amazing!", "Perfect!"
        ])
    reviews.append(base_review)
    labels.append(1)

# Add 500 negative reviews with variations
for i in range(500):
    base_review = np.random.choice(negative_reviews)
    # Add some natural variation
    if np.random.random() > 0.7:
        base_review = base_review + " " + np.random.choice([
            "Very disappointed.", "Not recommended.", "Poor quality.",
            "Waste of money.", "Terrible!", "Don't buy!"
        ])
    reviews.append(base_review)
    labels.append(0)

# Create DataFrame
df = pd.DataFrame({
    'text': reviews,
    'label': labels
})

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
df.to_csv('sample_dataset.csv', index=False)

print("Dataset created successfully!")
print(f"Total samples: {len(df)}")
print(f"Positive samples: {(df['label'] == 1).sum()}")
print(f"Negative samples: {(df['label'] == 0).sum()}")
print("\nFirst few rows:")
print(df.head(10))
print("\nDataset saved to: sample_dataset.csv")
