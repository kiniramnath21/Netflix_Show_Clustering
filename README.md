# 🎬 Netflix Show Clustering using Machine Learning

A data-driven unsupervised learning project that clusters Netflix shows and movies based on similarities in genre, cast, rating, and description.  
This project demonstrates end-to-end **Machine Learning model development, visualization, and deployment** using AWS EC2 and Streamlit.

---

## 🚀 Live Demo

🔗 **Deployed on AWS EC2:** [Netflix Show Clustering App](http://3.111.53.247:8501)

---

## 🧠 Project Overview

With thousands of shows and movies available on Netflix, identifying similar content manually is challenging.  
This project applies **K-Means Clustering** on the Netflix dataset to group shows with similar attributes, enabling users to explore clusters and visualize data relationships interactively.

---

## 🧩 Key Objectives

- Clean and preprocess the Netflix dataset.
- Perform feature extraction using **TF-IDF Vectorization** on textual descriptions.
- Apply **K-Means Clustering** to categorize shows and movies.
- Reduce dimensions with **PCA** for visual representation.
- Build an interactive **Streamlit web application**.
- Deploy the project using **AWS EC2** for public access.

---

## 🧰 Tech Stack

**Languages:** Python  
**Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn, plotly, nltk  
**Visualization & UI:** Streamlit, OxyPlot (for optional visualizations)  
**Deployment:** AWS EC2, pipenv / venv, nohup  
**Tools:** Jupyter Notebook, Visual Studio Code  

---

## 📊 Dataset

- **Source:** Netflix Movies and TV Shows dataset (Kaggle)  
- **Size:** ~8,800 records  
- **Features Used:** Title, Type, Director, Cast, Country, Release Year, Rating, Duration, Genre, Description  
- **Cleaning Performed:**
  - Removal of null and duplicate records  
  - Text normalization (lowercasing, punctuation removal, lemmatization)  
  - Tokenization and TF-IDF vectorization for description column  

---

## 🧮 Methodology

### 1. **Data Preprocessing**
- Loaded dataset and handled missing values.
- Extracted relevant columns for clustering.
- Preprocessed text data using **NLTK** for tokenization and stopword removal.

### 2. **Feature Extraction**
- Converted textual descriptions into numerical vectors using **TF-IDF**.

### 3. **Model Training**
- Applied **K-Means Clustering** with optimal `k` determined using **Elbow Method**.
- Grouped content into clusters based on similarity metrics.

### 4. **Dimensionality Reduction**
- Used **PCA (Principal Component Analysis)** for 2D visualization of clusters.

### 5. **Visualization**
- Visualized clusters and distributions using **Plotly Scatterplots** and **Matplotlib**.

### 6. **Web Application**
- Built an interactive **Streamlit dashboard**:
  - View cluster insights  
  - Search by show name  
  - Explore similar shows  
  - View 2D PCA visualization  

---

## 🌐 Deployment (AWS EC2)

The app was deployed on an **Amazon EC2 Ubuntu instance** using the following steps:

1. **Launched EC2 Instance**
   - Ubuntu 22.04, t2.micro (Free Tier)
   - Configured inbound rules for port **8501**

2. **Installed Dependencies**
   ```bash
   sudo apt update
   sudo apt install python3-pip
   pip install virtualenv

3. **Setup Virtual Environment**
   ```bash
   virtualenv venv
   source venv/bin/activate
   pip install -r requirements.txt

4. **Run Streamlit App in Background**
   ```bash
   nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &

5. **Access Application**

   Public URL: http://3.111.53.247:8501

## 📈 Results

    Shows and movies grouped into 5 distinct clusters.
    
    Visualization shows meaningful grouping based on genre, rating, and description.
    
    App provides real-time similarity search and interactive plots.

## 🔍 Example Output

    | Show Title      | Cluster | Similar Shows                        |
    | --------------- | ------- | ------------------------------------ |
    | Stranger Things | 3       | Dark, The OA, The Society            |
    | Money Heist     | 2       | Breaking Bad, Narcos, Ozark          |
    | The Crown       | 1       | The Queen, Victoria, The Royal House |

## 🔧 Installation (Run Locally)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/Netflix_Show_Clustering.git
   cd Netflix_Show_Clustering

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

4. **Run the application:**
   ```bash
   streamlit run app.py

5. **Access at:**

   👉 http://localhost:8501

## 🚀 Future Enhancements

- Integrate content recommendation system using cosine similarity.

- Add genre-based filtering and user ratings analysis.

- Deploy with Docker + AWS ECS or ECR for scalable cloud infrastructure.

- Integrate Hugging Face embeddings for advanced text similarity.

## 👨‍💻 Author
K. Ramanath

💼 Software & Machine Learning Engineer

📍 Bengaluru, India

🔗 LinkedIn

📧 kiniramnath21@gmail.com

## ⭐ If you found this project interesting, consider giving it a star!
    




