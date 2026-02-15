import pandas as pd
import numpy as np
import random

# Set random seed
np.random.seed(42)
random.seed(42)

# News categories and templates
categories = {
    0: "Technology",
    1: "Sports", 
    2: "Business",
    3: "Entertainment"
}

# Technology news templates
tech_news = [
    "Apple announces new iPhone with revolutionary AI features and improved battery life.",
    "Google launches advanced machine learning platform for developers worldwide.",
    "Microsoft acquires gaming company in landmark $70 billion deal.",
    "Tesla unveils latest electric vehicle with autonomous driving capabilities.",
    "Meta introduces new virtual reality headset for immersive experiences.",
    "Amazon develops AI assistant that can understand complex commands.",
    "Researchers achieve breakthrough in quantum computing technology.",
    "New smartphone app uses AI to detect health conditions early.",
    "Tech giant invests billions in renewable energy infrastructure.",
    "Cybersecurity experts warn about new malware targeting businesses.",
    "Social media platform introduces enhanced privacy features for users.",
    "Scientists develop faster processors using advanced chip technology.",
    "Streaming service launches new feature powered by artificial intelligence.",
    "Major tech company announces plans to hire thousands of engineers.",
    "Startup raises millions to develop revolutionary battery technology.",
]

# Sports news templates
sports_news = [
    "Local team wins championship after thrilling overtime victory.",
    "Star athlete signs record-breaking contract worth millions.",
    "National soccer team advances to finals after stunning performance.",
    "Tennis champion claims another Grand Slam title in epic match.",
    "Basketball legend announces retirement after illustrious career.",
    "Olympic committee reveals host city for upcoming games.",
    "Marathon runner breaks world record at international competition.",
    "Baseball team makes historic comeback in ninth inning.",
    "Football coach receives award for outstanding season performance.",
    "Gymnast wins gold medal with perfect score at championships.",
    "Hockey team celebrates victory with hometown parade.",
    "Swimmer qualifies for Olympics with personal best time.",
    "Young athlete becomes youngest champion in sport's history.",
    "Cricket match ends in dramatic tie-breaking finale.",
    "Golf tournament sees upset victory by underdog player.",
]

# Business news templates
business_news = [
    "Stock market reaches record high amid strong economic indicators.",
    "Major retailer announces expansion plans with hundreds of new stores.",
    "Central bank adjusts interest rates to control inflation.",
    "Startup company goes public with successful IPO launch.",
    "Pharmaceutical giant develops breakthrough medication for disease.",
    "Oil prices surge following international production cuts.",
    "Real estate market shows signs of recovery in major cities.",
    "Automotive manufacturer reports record quarterly earnings.",
    "Economic growth exceeds expectations in latest report.",
    "Trade agreement signed between major economic powers.",
    "Consumer spending increases during holiday shopping season.",
    "Banking sector implements new digital payment system.",
    "Company announces layoffs amid restructuring efforts.",
    "Merger creates industry giant in telecommunications sector.",
    "Investors show confidence with strong market performance.",
]

# Entertainment news templates
entertainment_news = [
    "Blockbuster movie breaks box office records on opening weekend.",
    "Famous actor wins prestigious award for outstanding performance.",
    "New album debuts at number one on music charts worldwide.",
    "Popular TV series announces final season to fans' disappointment.",
    "Celebrity couple announces engagement in social media post.",
    "Music festival attracts thousands of fans from around the world.",
    "Director unveils trailer for highly anticipated film sequel.",
    "Broadway show receives standing ovation at premiere night.",
    "Streaming platform releases hit series that captivates audiences.",
    "Rock band announces reunion tour after decade-long hiatus.",
    "Award show celebrates best performances of the year.",
    "Famous author releases bestselling novel to critical acclaim.",
    "Animation studio reveals plans for upcoming feature film.",
    "Pop star performs sold-out concert at major venue.",
    "Reality show returns for new season with fresh contestants.",
]

# Generate dataset
news_items = []
labels = []
category_names = []

# Generate 500 samples per category (2000 total)
for label, category in categories.items():
    templates = [tech_news, sports_news, business_news, entertainment_news][label]
    
    for i in range(500):
        # Select random template
        news = random.choice(templates)
        
        # Add some variation
        if random.random() < 0.2:
            news = news + " Industry experts are closely monitoring the situation."
        
        news_items.append(news)
        labels.append(label)
        category_names.append(category)

# Create DataFrame
df = pd.DataFrame({
    'text': news_items,
    'label': labels,
    'category': category_names
})

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Add ID
df.insert(0, 'id', range(1, len(df) + 1))

# Save
output_path = 'data/raw/news_classification_2000.csv'
df.to_csv(output_path, index=False)

print("=" * 70)
print("MULTI-CLASS NEWS DATASET CREATED!")
print("=" * 70)
print(f"\n📊 Dataset Statistics:")
print(f"   Total samples: {len(df)}")
print(f"   Number of classes: {df['label'].nunique()}")

for label, category in categories.items():
    count = (df['label'] == label).sum()
    print(f"   {category} (label={label}): {count} samples")

print(f"\n💾 File saved to: {output_path}")

print(f"\n🔍 Sample news from each category:")
print("-" * 70)
for label in range(4):
    sample = df[df['label'] == label].iloc[0]
    print(f"\n[{sample['category']}]")
    print(f"{sample['text']}")

print("\n" + "=" * 70)
print("✅ Use with: python main.py --data data/raw/news_classification_2000.csv")
print("=" * 70)
