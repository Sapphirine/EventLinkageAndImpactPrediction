from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from tqdm import tqdm
from fuzzywuzzy import process
import spacy        

nlp = spacy.load('en_core_web_lg')
analyzer = SentimentIntensityAnalyzer()

df = pd.read_csv('./S&P.csv', index_col=0)

stockDic = {}
stockNameList = []
stockSymbolList = []
matrix = np.load('matrix.npy')
U, sigma, VT = np.linalg.svd(matrix)
# print('U=', U)
# print('sigma=', sigma)
# print('VT=', VT)
sig2 = []
feature = 96
for i in range(feature):
    z = [0] * feature
    z[i] = sigma[i]
    sig2.append(z)
sig2 = np.mat(sig2)
matrix = U[:,:feature] * sig2 * VT[:feature,:]
matrix /= max(matrix.max(1).tolist())

for i in range(len(df)):
    stockName = df['Security'][i].lower()
    stockSymbol = df['Symbol'][i].lower()
    stockNameList.append(stockName)
    stockSymbolList.append(stockSymbol)
    stockDic[stockName] = stockSymbol

def getMention(document_string):
    sents = nlp(document_string)
    companyNames = [str(ee) for ee in sents.ents if ee.label_ == 'ORG']
    res = []
    for company in tqdm(companyNames):
        ratios = process.extract(company, stockNameList)
        res.append(stockDic[ratios[0][0]])
    return companyNames, res

def impactCompanys(symbols, score):
    res = []
    for symbol in symbols:
        idx = stockSymbolList.index(symbol)
        res2 = []
        big5 = matrix[idx,:].argsort().tolist()[0][::-1][1:6]
        print(big5)
        for x in big5:
            res.append((stockNameList[x], matrix[idx, x] * score))
    res = sorted(res, key=lambda x: x[1], reverse=True)
    return res

def getScore(text):
    score = analyzer.polarity_scores(text)['compound']
    companyNames, symbols = getMention(text)
    return companyNames, symbols, score, impactCompanys(symbols, score)  

from flask import Flask, render_template, request
app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('base.html')
    
@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    a, b, c, d = getScore(text)
    return {'Mentioned Company': a, 'Mentioned Symbol': b, 'Impact Score': c, 'Other Impact Companies': d}
    
if __name__== '__main__':
    app.run(debug=True, host="0.0.0.0", port= 8787)