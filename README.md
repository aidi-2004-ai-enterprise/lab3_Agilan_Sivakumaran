# Lab 3 - Penguins Classifier with XGBoost and FastAPI 🐧

This is a fun project where I built a machine learning model to guess the type of penguin (species) based on some of its features. I used a dataset from the Seaborn library and trained a model using XGBoost. Then, I created a FastAPI app so you can send penguin info and get a prediction.

# Demo Video (Uploaded in this repo as well)
lab3_Agilan_Sivakumaran/Lab3-20250724_182243-Meeting Recording.mp4

---

## 🔧 How to Set It Up

### 1. Clone the Repo
First, download the project to your computer.

git clone https://github.com/aidi-2004-ai-enterprise/lab3_your_firstname_your_lastname.git
cd lab3_your_firstname_your_lastname

### 2. Install uv (if you don’t already have it)
pip install uv

### 3. Install Dependencies
uv pip install -r pyproject.toml

### 4. Train the Model
python train.py

This will save the trained model file to:
app/data/model.json

### 5. Start the API
uvicorn main:app --reload
http://127.0.0.1:8080/docs

### 6. Health Check
http://127.0.0.1:8080/health
You should see: { "status": "ok" }

