# 📚 BookLab — Popularity Predictor & Recommender

**BookLab** is a sleek, interactive Streamlit app that predicts book popularity, recommends similar titles, and provides rich insights on authors, publishers, and ratings — all with dynamic cover images.  

---

## Features

- 🔮 **Predict Book Ratings & Popularity**: Enter a book title and get predicted ratings and popularity.  
- 📖 **Explore Books**: Browse the top books with filters for language, publisher, and page count.  
- 🌟 **Recommendations**: See similar books with cover images and similarity scores.  
- 📊 **Insights Dashboard**:  
  - Top authors by influence  
  - Top publishers by influence  
  - Rating distribution  
  - Popularity breakdown  
  - Feature importance for predictions  
- ⚡ Fully optimized for performance and quick load with caching.  

---

## Screenshots

<img width="1887" height="878" alt="image" src="https://github.com/user-attachments/assets/d398af36-f709-44b0-93d7-49f60262107a" />
 
<img width="1847" height="845" alt="image" src="https://github.com/user-attachments/assets/edc3bdbd-ae22-402b-bf0c-b9330476a692" />

<img width="1835" height="907" alt="image" src="https://github.com/user-attachments/assets/0947f577-ffc8-4716-9ea3-54b945635cae" />



---

## Installation
Clone the repository:
   ```bash
   git clone https://github.com/yourusername/booklab.git
   cd booklab
   ```

## Getting Started

### 1. Download the Dataset

The dataset is derived from Goodreads and used for predictions. You can download it here:  
[Kaggle Dataset Link](https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks)  

Save the CSV file (`books_cleaned_featured.csv`) in the project folder.

---

### 2. Run the Project

Everything is in the Jupyter notebook `projectB.ipynb`:

1. Open `projectB.ipynb` in **Jupyter Notebook** or **JupyterLab**.
2. Run all cells sequentially — it contains all imports, preprocessing, model loading, and Streamlit app code.
3. Start the Streamlit app from the notebook or terminal:
   ```bash
   streamlit run app.py
   ```

---

### 3. Dataset & Models

Dataset: Goodreads-derived dataset (included as books_cleaned_featured.csv)

ML Models: Random Forest Regressor and Classifier (will be stored in models folder)

<img width="316" height="249" alt="image" src="https://github.com/user-attachments/assets/c4edf582-66f2-4f38-b967-81a247f462b0" />



---

### 4. Usage


Open the app in your browser.

Use the sidebar to filter books by language, publisher, or page count.

Search a specific book title to get predictions and recommendations.

Explore top authors, publishers, ratings, and popularity in the dashboard.

---

### 5.Credits

Data: Goodreads

Covers: Google Books API & Open Library

Built with ❤️ using Streamlit, Pandas, NumPy, Scikit-learn, and Plotly

