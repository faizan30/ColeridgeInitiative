# A project for Coleridge Initiative- show US the data on kaggle 
https://www.kaggle.com/faizan86/notebook2819b5f158/notebook

# create a virtual environment
virtualenv -p python3.7 env
source env/bin/activate

# install requirements
pip install -r requirements.txt

# download Spacy data
python -m spacy download en

# download NLTK data
python -c "import nltk; nltk.download('punkt')"