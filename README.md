📩 SMS Spam Detector (Streamlit App)

A simple and interactive SMS Spam Detection Web App built using Python, Scikit-Learn, TF-IDF, and Streamlit.
The model predicts whether a given SMS message is Spam or Ham (Not Spam) based on text classification.

🚀 Features

🔤 Text Input — Type or paste any SMS message

🤖 ML Model Prediction — RandomForest + TF-IDF

📊 Spam Probability Score

🧠 Top Important Keywords Table

🧪 Example Messages for Quick Testing

⚡ Fast, Lightweight, and Easy to Deploy

🌐 Works fully on Streamlit Cloud / Local Machine

🛠️ Tech Stack

Python 3.x

Streamlit

Scikit-Learn

NumPy / Pandas

Pickle for model + vectorizer loading

📁 Project Structure
sms-spam-detector/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── spam_sms.csv   (optional, training dataset)
└── README.md

▶️ Run the App Locally
1. Clone the repository
git clone https://github.com/your-username/sms-spam-detector.git
cd sms-spam-detector

2. Install dependencies
pip install -r requirements.txt

3. Run the Streamlit app
streamlit run app.py


Once started, Streamlit will show:

Local URL: http://localhost:8501


Open it in your browser.

📦 Requirements

Contents of requirements.txt:

streamlit==1.51.0
scikit-learn==1.3.2
numpy==1.26.5
pandas==2.3.3

📊 Model Details

The ML pipeline includes:

Vectorizer: TF-IDF text vectorization

Model: RandomForestClassifier

Training: Model trained on SMS spam dataset

Pickle Files:

model.pkl → trained classifier

vectorizer.pkl → fitted TF-IDF transformer

🌐 Deployment Guide (Streamlit Cloud)

Create a GitHub repository

Push all files including model.pkl, vectorizer.pkl, requirements.txt, and app.py

Go to https://share.streamlit.io

Connect your GitHub repo

Deploy 🚀

📸 App Preview

Add a screenshot here if you want
Example:

![App Screenshot](screenshot.png)

🤝 Contributing

Feel free to fork this project and improve it! PRs are welcome.

📄 License

This project is open-source under the MIT License.
