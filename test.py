import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

url0 = 'https://www.linkedin.com/jobs/search/?f_E=2&geoId=101282230&keywords=the%20data%20science&location=Germany'
pages = list(range(25,1000,25))
url_list=[url0]
links_text = []
websites = ['https://www.linkedin.com/in/diego-barra-ureta/']

# LOAD CV AND EXTRACT TEXT


path = r'C:\Users\Diego\Downloads\THESIS\After thesis\Vorlage Erika Mustermann\CV_Diego_Barra.pdf'
path0 = r'C:\Users\Diego\Downloads\Salma_CV_2019-3.pdf'

# For extracting the CV text from pdf file

fp = open(path, 'rb')
from pdfminer.pdfparser import PDFParser, PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine

parser = PDFParser(fp)
doc = PDFDocument()
parser.set_document(doc)
doc.set_parser(parser)
doc.initialize('')
rsrcmgr = PDFResourceManager()
laparams = LAParams()
device = PDFPageAggregator(rsrcmgr, laparams=laparams)
interpreter = PDFPageInterpreter(rsrcmgr, device)
cv_text = ''

for page in doc.get_pages():
    interpreter.process_page(page)
    layout = device.get_result()
    for lt_obj in layout:
        if isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextLine):
            cv_text += lt_obj.get_text()


cv_text = cv_text.lower().replace('[^a-zA-Z]', ' ').replace('.','').replace(',','').replace(':','')
links_text.append(cv_text)

with requests.Session() as s:
    response = s.get(url0)
    html = response.content
    soup = BeautifulSoup(html, 'html.parser')
    links = soup.findAll('a', {'class': "result-card__full-card-link"})

for a in links:
    with requests.Session() as s:
        website = a.get('href')
        websites.append(website)
        response = s.get(website)
        html_ = response.content
        soup_ = BeautifulSoup(html_, 'html.parser')
        raw = soup_.find('div', {'class': 'description__text description__text--rich'})
        text = raw.get_text()
        links_text.append(text)

for p in pages:
    url = url0+'&start='+str(p)
    url_list.append(url)
    with requests.Session() as s:
        response = s.get(url)
        html = response.content
        soup = BeautifulSoup(html, 'html.parser')
        links = soup.findAll('a', {'class' : "result-card__full-card-link"})

    for a in links:
        with requests.Session() as s:
            website = a.get('href')
            websites.append(website)
            response = s.get(website)
            html_= response.content
            soup_= BeautifulSoup(html_,'html.parser')
            raw = soup_.find('div', {'class': 'description__text description__text--rich'})
            text = raw.get_text()
            links_text.append(text)

# Getting the raw text dataframe with the job links.


links_text_array = np.asarray(links_text)
df = pd.DataFrame(data = links_text_array,columns=['text'])
df = df.text.str.replace('[^a-zA-Z]', ' ').str.lower()
print(df.head())
print(df.shape)

# Apply count vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


cv = TfidfVectorizer(max_features=300,stop_words='english',min_df=0.2,max_df=0.8,ngram_range=(2,3))
cv_ = cv.fit_transform(links_text)
cos_similarity = cosine_similarity(cv_,cv_)
cos_sim_col = cos_similarity[0]


print("cosine similarity matrix:")
print(cos_similarity)
cols = cv.get_feature_names()

# Generate dataframe with the CountVectorizer model
cv_df = pd.DataFrame(cv_.toarray(),columns=cols).add_prefix('Counts_')
cv_df['websites'] = pd.Series(websites)
cv_df['cos_similarity'] =pd.Series(cos_sim_col)
print('CountVectorizer dataframe:')
cv_df = cv_df.sort_values(by='cos_similarity',ascending=False)
cv_df = cv_df.drop_duplicates(['cos_similarity'])
print("matching websites: ")
pd.set_option('display.max_colwidth', -1)
print(cv_df.websites.head(50))
print("length of websites list: " + str(len(websites)))
print("shape of cv_df: " +str(cv_df.shape))

# Apply TfidfVectorizer

tv = TfidfVectorizer(max_features=300,stop_words='english',min_df=0.2,max_df=0.8,ngram_range=(2,3))
tv_transformed = tv.fit_transform(df)

# Generate dataframe with the model

tv_df = pd.DataFrame(tv_transformed.toarray(), columns = tv.get_feature_names()).add_prefix('TFIDF_')
sample_row = tv_df.iloc[0]
print('TfidfVectorizer dataframe:')
print(sample_row.sort_values(ascending=False).head(50))

results = sample_row.sort_values(ascending=False).head(50)
print(results)
