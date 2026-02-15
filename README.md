# AI Final Project - Text Classification

## 📋 Project Overview
This project implements a complete text classification system using Natural Language Processing (NLP) and Machine Learning techniques. The goal is to classify text documents into predefined categories with high accuracy.

**Course:** Artificial Intelligence  
**Instructor:** Dr. Pishgoo  
**Project Supervisor:** Eng. Alireza Ghorbani

## 🎯 Problem Definition
The objective is to build an intelligent text classification system that can accurately categorize text documents. The model aims to achieve:
- **Accuracy:** ≥ 85%
- **F1-Score:** ≥ 0.80

## 📊 Dataset
- **Source:** [Specify dataset source, e.g., IMDB, 20 Newsgroups, AG News, etc.]
- **Size:** [Number of samples]
- **Classes:** [Number of categories]
- **Type:** Text data

## 🛠️ Technologies Used
- **Programming Language:** Python 3.9+
- **ML Frameworks:** scikit-learn, PyTorch, Transformers
- **NLP Libraries:** NLTK, Hugging Face
- **Visualization:** Matplotlib, Seaborn, Plotly

## 📁 Project Structure
```
AI-FinalProject-TextClassification/
├── data/
│   ├── raw/              # Original dataset
│   └── processed/        # Preprocessed data
├── notebooks/
│   ├── 01_EDA.ipynb     # Exploratory Data Analysis
│   └── 02_experiments.ipynb  # Model experiments
├── src/
│   ├── preprocessing/    # Data preprocessing modules
│   ├── models/          # Model architectures
│   ├── training/        # Training scripts
│   ├── evaluation/      # Evaluation metrics
│   └── utils/           # Helper functions
├── results/
│   ├── charts/          # Visualizations
│   └── metrics/         # Performance metrics
├── requirements.txt     # Dependencies
├── README.md           # This file
└── .gitignore          # Git ignore rules
```

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/[username]/AI-FinalProject-TextClassification.git
cd AI-FinalProject-TextClassification
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK data
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

## 📈 Usage

### Data Preprocessing
```bash
python src/preprocessing/text_processor.py
```

### Training Baseline Model
```bash
python src/training/trainer.py --model baseline --epochs 10
```

### Training Advanced Model (BERT)
```bash
python src/training/trainer.py --model bert --epochs 5 --batch_size 16
```

### Evaluation
```bash
python src/evaluation/evaluator.py --model_path results/models/best_model.pt
```

### Run Demo
```bash
streamlit run demo/app.py
```

## 🧪 Experiments

### Phase 1: Baseline Models
1. **Logistic Regression** with TF-IDF
2. **Naive Bayes** 
3. **Random Forest**

### Phase 2: Advanced Models
1. **CNN for Text**
2. **LSTM/BiLSTM**
3. **BERT (fine-tuned)**

## 📊 Results

### Baseline Model Performance
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.XX | 0.XX | 0.XX | 0.XX |
| Naive Bayes | 0.XX | 0.XX | 0.XX | 0.XX |
| Random Forest | 0.XX | 0.XX | 0.XX | 0.XX |

### Final Model Performance
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| BERT | 0.XX | 0.XX | 0.XX | 0.XX |

## 🔧 Improvements Applied
1. **Data Augmentation:** Back-translation, synonym replacement
2. **Hyperparameter Tuning:** Learning rate, batch size optimization
3. **Transfer Learning:** Fine-tuned BERT model
4. **Regularization:** Dropout, early stopping


## 📝 References
1. [Paper/Article 1]
2. [Paper/Article 2]
3. [Dataset Source]

## 📄 License
This project is for educational purposes only.

## 🙏 Acknowledgments
- Dr. Pishgoo for course instruction
- Eng. Alireza Ghorbani for project supervision
- [Any other acknowledgments]
