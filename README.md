# News Source Classification using RoBERTa

## About the Project
This project focuses on binary classification of news headlines, distinguishing between articles from **Fox News** and **NBC News**. We compare traditional machine learning models (TF-IDF, GloVe) with transformer-based models (RoBERTa), and show how contextual embeddings improve accuracy and generalization.

## Key Features & Modeling Pipeline

- Collected 3,800+ news headlines using web scraping
- Cleaned and preprocessed text (e.g., decontractions, punctuation removal, lowercasing)
- Built baseline models using TF-IDF features:
  - Logistic Regression, Naive Bayes, and Linear SVM
- Integrated **GloVe embeddings** for improved classical model performance
- Fine-tuned **RoBERTa model** using Hugging Face Transformers
- Final RoBERTa model achieved **86.15%** accuracy on test set

## Repository Structure
The repository includes:
- `News_Classification_NLP.ipynb` — full notebook with scraping, preprocessing, modeling, and evaluation  
- `News_Classification_NLP_Report.pdf` — slides summarizing project process and findings

## Dependencies
Install required packages:
```bash
pip install transformers torch pandas numpy scikit-learn spacy
python -m spacy download en_core_web_sm
```

## To Run the Code
Open the notebook:
```bash
jupyter notebook News_Classification_NLP.ipynb
```
Or run specific model pipelines (e.g., GloVe or RoBERTa) in Colab or your local environment.

## Model Access
Our fine-tuned RoBERTa model is hosted on Hugging Face:  
🔗 https://huggingface.co/CIS5190GoGo/CustomModel

You can load the model using:
```python
from transformers import RobertaForSequenceClassification, RobertaTokenizer

model = RobertaForSequenceClassification.from_pretrained("CIS5190GoGo/CustomModel")
tokenizer = RobertaTokenizer.from_pretrained("CIS5190GoGo/CustomModel")
```

## Collaborators
Jiayi Chen, Feng Jiang, Zihan Wang
