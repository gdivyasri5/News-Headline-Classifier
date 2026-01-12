# 📰 News Headline Classification AI

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![Library](https://img.shields.io/badge/Library-Scikit--Learn-orange?style=flat)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
This project is a **Natural Language Processing (NLP)** application that automatically categorizes news headlines into domains such as **Sports, Politics, Technology, Finance, and Entertainment**. 

It utilizes the **Naive Bayes** algorithm to learn from textual data and predict the category of new, unseen headlines with high accuracy. This tool demonstrates the practical application of Supervised Machine Learning in information filtering.

## 🚀 Key Features
- **Intelligent Prediction:** Uses a probabilistic model (`MultinomialNB`) to classify text.
- **Text Vectorization:** Converts raw text into numerical data using `CountVectorizer`.
- **Real-Time Interaction:** Includes a CLI (Command Line Interface) for instant user testing.
- **Lightweight:** Entire logic contained in a single, efficient script.

## 🛠️ Tech Stack
- **Language:** Python 3
- **Core Logic:** Scikit-learn (Sklearn)
- **Data Handling:** Pandas
- **Serialization:** Pickle (ready for model saving)

## 📂 How to Run Locally

### 1. Prerequisites
Ensure you have Python installed. Then, install the required libraries:
```bash
pip install pandas scikit-learn
