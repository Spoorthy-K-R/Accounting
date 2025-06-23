

import os
import pandas as pd
import nltk
import string

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# finance_keywords = [ ... ]  # your list
lm_dict = pd.read_csv('Loughran-McDonald_MasterDictionary_1993-2024.csv')

# 2. Extract word lists for categories (e.g., Positive, Negative, Uncertainty)
positive_words = set(lm_dict[lm_dict['Positive'] > 0]['Word'].str.lower())
negative_words = set(lm_dict[lm_dict['Negative'] > 0]['Word'].str.lower())
uncertainty_words = set(lm_dict[lm_dict['Uncertainty'] > 0]['Word'].str.lower())
# ...add more categories as needed

# 3. Combine or use separately
finance_words = positive_words | negative_words | uncertainty_words  # union of sets


stop_words = set(stopwords.words('english'))

# 1. Read company mapping
company_df = pd.read_csv('sp100_list.csv', dtype=str)
# Assume columns: 'CIK', 'Company Name'
cik_to_company = dict(zip(company_df['cik'].str.zfill(10), company_df['conm']))

def preprocess(sentence):
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation + '0123456789'))
    tokens = word_tokenize(sentence)
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

def contains_finance_word(sentence, finance_words):
    tokens = set(word_tokenize(sentence.lower()))
    return bool(tokens & finance_words) 

results = []

for fname in os.listdir('/Users/spoorthy/Projects/Accounting/10K'):
    if fname.endswith('.txt'):
        cik = fname.split('-')[0].zfill(10)  # Extract and pad CIK
        company = cik_to_company.get(cik, 'Unknown')
        with open(os.path.join('10K', fname), 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            sentences = sent_tokenize(text)
            for sent in sentences:
                found = [kw for kw in finance_words if kw in sent.lower()]
                if len(found)>0:
                    cleaned = preprocess(sent)
                    results.append({
                        'company': company,
                        'file': fname,
                        'original_sentence': sent,
                        'cleaned_sentence': cleaned,
                        'matched_keywords': ', '.join(found)
                    })

df = pd.DataFrame(results)

# 5. Store grouped by company (CSV)
df.to_csv('finance_sentences_by_company.csv', index=False)

# 6. (Optional) Store as JSON grouped by company
grouped = df.groupby('company').apply(lambda x: x.to_dict(orient='records')).to_dict()
import json
with open('finance_sentences_by_company.json', 'w') as f:
    json.dump(grouped, f, indent=2)