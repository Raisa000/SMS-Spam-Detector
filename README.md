# SMS Spam Detector (Streamlit App)

A simple and interactive SMS Spam Detection Web App built using Python, Scikit-Learn, TF-IDF, and Streamlit.  
The model predicts whether a given SMS message is Spam or Ham (Not Spam).

------------------------------------------------------------
FEATURES
------------------------------------------------------------
- Text input box for SMS messages
- RandomForest + TF-IDF based prediction
- Shows spam probability score
- Displays top important keywords
- Example messages for testing
- Runs smoothly on Streamlit Cloud and locally

------------------------------------------------------------
TECH STACK
------------------------------------------------------------
- Python 3.x
- Streamlit
- Scikit-Learn
- NumPy / Pandas
- Pickle (for loading model & vectorizer)

------------------------------------------------------------
PROJECT STRUCTURE
------------------------------------------------------------
sms-spam-detector/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── spam_sms.csv        # optional dataset
└── README.md


------------------------------------------------------------
RUN LOCALLY
------------------------------------------------------------
1. Clone the repository:
   git clone https://github.com/your-username/sms-spam-detector.git
   cd sms-spam-detector

2. Install dependencies:
   pip install -r requirements.txt

3. Run the app:
   streamlit run app.py

Streamlit will open the app at:
http://localhost:8501

------------------------------------------------------------
REQUIREMENTS.TXT (copy this)
------------------------------------------------------------
streamlit==1.51.0
scikit-learn==1.3.2
numpy==1.26.5
pandas==2.3.3

------------------------------------------------------------
MODEL DETAILS
------------------------------------------------------------
- Vectorizer: TF-IDF
- Model: RandomForestClassifier
- Training: Standard SMS Spam dataset
- Files produced:
  - model.pkl (trained classifier)
  - vectorizer.pkl (TF-IDF transformer)

------------------------------------------------------------
DEPLOY ON STREAMLIT CLOUD
------------------------------------------------------------
1. Push all files to GitHub (including .pkl files)
2. Visit https://share.streamlit.io
3. Connect your GitHub repo
4. Select main branch
5. Deploy

------------------------------------------------------------
CONTRIBUTING
------------------------------------------------------------
Feel free to fork this project and improve it.
Pull requests are welcome.

------------------------------------------------------------
LICENSE
------------------------------------------------------------
This project is open-source under the MIT License.
