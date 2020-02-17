from flask import Flask,render_template,request
from werkzeug.utils import secure_filename

from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import io

import spacy
from spacy.matcher import Matcher
import re
import pandas as pd
import spacy
from nltk.tokenize import word_tokenize
from nltk.util import ngrams


def extract_text_from_pdf(pdf_path):
    page = ""
    with open(pdf_path, 'rb') as fh:
        # iterate over all pages of PDF document
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            # creating a resoure manager
            resource_manager = PDFResourceManager()
            
            # create a file handle
            fake_file_handle = io.StringIO()
            
            # creating a text converter object
            converter = TextConverter(
                                resource_manager, 
                                fake_file_handle, 
                                codec='utf-8', 
                                laparams=LAParams()
                        )

            # creating a page interpreter
            page_interpreter = PDFPageInterpreter(
                                resource_manager, 
                                converter
                            )

            # process current page
            page_interpreter.process_page(page)
            
            # extract text
            text = fake_file_handle.getvalue()
            yield text

            # close open handles
            converter.close()
            fake_file_handle.close()

import docx2txt

def extract_text_from_doc(doc_path):
    temp = docx2txt.process(doc_path)
    text = [line.replace('\t', ' ') for line in temp.split('\n') if line]
    return ' '.join(text)

nlp = spacy.load('./model5')

def extract_name(resume_text):
    matcher = Matcher(nlp.vocab)
    nlp_text = nlp(resume_text)

    # First name and Last name are always Proper Nouns
    pattern = [[{'POS': 'PROPN'}, {'POS': 'PROPN'}]]
    
    matcher.add('NAME', None, *pattern)
    
    matches = matcher(nlp_text)
    
    for match_id, start, end in matches:
        span = nlp_text[start:end]
        return [span.text]


def extract_mobile_number(text):
    phone = re.findall(re.compile(r'(?:(?:\+?([1-9]|[0-9][0-9]|[0-9][0-9][0-9])\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([0-9][1-9]|[0-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?'), text)

    if phone:
        number = ''.join(phone[0])
        if len(number) > 10:
            return ['+' + number]
        else:
            return [number]


def extract_email(email):
    email = re.findall("([^@|\s]+@[^@]+\.[^@|\s]+)", email)
    if email:
        try:
            return [email[0].split()[0].strip(';')]
        except IndexError:
            return None


def convertTuple(tup): 
    str =  ' '.join(tup) 
    return str

def extract_skills(resume_text):
    nlp_text = nlp(resume_text)

    # removing stop words and implementing word tokenization
    tokens = [token.text for token in nlp_text if not token.is_stop]
    tokens = word_tokenize(resume_text)
    bigrams = ngrams(tokens,2)
    trigrams = ngrams(tokens,3)
        
    # reading the csv file    
    skills = []
    with open("skills.txt") as file_in:
        for line in file_in:
            skills.append(line.replace("\n", ""))
                
    skillset = []
    # check for one-grams (example: python)
    for token in tokens:
        if token in skills:
            skillset.append(token)
        elif token.lower() in skills:
            skillset.append(token)
        elif token.upper() in skills:
            skillset.append(token)

            
    # check for bi-grams and tri-grams (example: machine learning)
    for token in bigrams:
        token = convertTuple(token)
        if token in skills:
            skillset.append(token)
        elif token.lower() in skills:
            skillset.append(token)
        elif token.upper() in skills:
            skillset.append(token)
                            
                
    for token in trigrams:
        token = convertTuple(token)
        if token in skills:
            skillset.append(token)
        elif token.lower() in skills:
            skillset.append(token)
        elif token.upper() in skills:
            skillset.append(token)
                            
    return [i.capitalize() for i in set([i.lower() for i in skillset])]


from nltk.corpus import stopwords


STOPWORDS = set(stopwords.words('english'))

# Education Degrees
EDUCATION = [
            'BACCALAUREATE', 'BACCALAUREAT', 'LICENCE PROFESSIONELLE', 'LICENCE FONDAMENTALE', 'LP', 'LF'
            'BE','B.E.', 'B.E', 'BS', 'B.S', 'BACHELOR', "BACHELOR'S", 'M'
            'ME', 'M.E', 'M.E.', 'MS', 'M.S', 'MASTER', "MASTER'S"
            'BTECH', 'B.TECH', 'M.TECH', 'MTECH', 'DUT', 'DEUG', 'BTS', 'DTS',
            'SSC', 'HSC', 'CBSE', 'ICSE', 'X', 'XII'
            ]

def extract_education(resume_text):
    nlp_text = nlp(resume_text)

    # Sentence Tokenizer
    nlp_text = [sent.string.strip() for sent in nlp_text.sents]

    edu = {}
    # Extract education degree
    for index, text in enumerate(nlp_text):
        for tex in text.split():
            # Replace all special symbols
            tex = re.sub(r'[?|$|.|!|,]', r'', tex)
            if tex.upper() in EDUCATION and tex not in STOPWORDS:
                edu[tex] = text + nlp_text[index + 1]

    # Extract year
    education = []
    for key in edu.keys():
        year = re.search(re.compile(r'(((20|19)(\d{2})))'), edu[key])
        if year:
            education.append((key, ''.join(year[0])))
        else:
            education.append(key)
    return education


def del_nones(dfObj):
    for col in dfObj:
        dfObj[col] = dfObj[col].astype(str).str.replace(u'None',u'--')
        dfObj[col] = dfObj[col].astype(str).str.replace(u'nan',u'--')
    return dfObj


def all_together(file_path):
    if file_path.endswith('.pdf'):
        for page in extract_text_from_pdf(file_path):
            page += ' '+page
    elif file_path.endswith('.docx'):
        page = extract_text_from_doc(file_path)
        
    name, phone, email, edu, skills = extract_name(page), extract_mobile_number(page), extract_email(page), extract_education(page), extract_skills(page)

    if name == None:
        name = '--'
    if phone == None:
        phone = '--'
    if email == None:
        email = '--'
    if edu == None:
        edu = '--'
    if skills == None:
        skills = '--'
    
    dfObj = pd.DataFrame([name, phone, email, edu, skills], index=['Name', 'Phone Number', 'Email Adress', 'Diploma', 'Skills']).T
    
    return del_nones(dfObj), page, [name, phone, email, edu, skills]


def ner_assist(df, page):
    doc_to_test = nlp(page)
    d={}
    entities = []
    for ent in doc_to_test.ents:
        d[ent.label_] = []
        entities.append(ent.label_)
    for ent in doc_to_test.ents:
        d[ent.label_].append(ent.text)

    title = []
    inst = []
    skills = []
    newlist = []

    if "TITLE" in entities:
        for val in set(d["TITLE"]):
            title.append(val)
    if "INSTITUTION" in entities:
        for val in set(d["INSTITUTION"]):
            inst.append(val)

    df['Diploma Title'] = pd.Series(title)
    df['Institution'] = pd.Series(inst)
    
    return del_nones(df)


def Analyse_CV_coverletter(cv, cover):
    df_cv, page_cv, listall = all_together(cv)
    df_cv = ner_assist(df_cv, page_cv)
    
    df_cover, page_cover, listall = all_together(cover)
    df_cover = ner_assist(df_cover, page_cover)
    return pd.concat([df_cv, df_cover])

app = Flask(__name__,template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/upload', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         f = request.files['file']
#         f2 = request.files['file2']
#         f.save('static/cv/' + secure_filename(f.filename))
#         f2.save('static/cover/' + secure_filename(f2.filename))
#         dataf =  Analyse_CV_coverletter('static/cv/' + secure_filename(f.filename), 'static/cover/' + secure_filename(f2.filename))
#     return render_template('results.html',  tables=[dataf.to_html(classes='data')], titles=dataf.columns.values)

# @app.route('/results', methods=['GET', 'POST'])
# def upload_file2():
#     if request.method == 'POST':
#         f = request.files['file']
#         f2 = request.files['file2']
#         f.save('static/cv/' + secure_filename(f.filename))
#         f2.save('static/cover/' + secure_filename(f2.filename))
#         dataf =  Analyse_CV_coverletter('static/cv/' + secure_filename(f.filename), 'static/cover/' + secure_filename(f2.filename))
#     return render_template('results.html',  tables=[dataf.to_html(classes='data')], titles=dataf.columns.values)

@app.route('/basic_elements')
def basic_elements():
    return render_template('basic_elements.html')

@app.route('/basic_elements', methods=['GET', 'POST'])
def basic_elements2():
    dataf = None
    if request.method == 'POST':
        f = request.files['file']
        f2 = request.files['file2']
        f.save('static/cv/' + secure_filename(f.filename))
        f2.save('static/cover/' + secure_filename(f2.filename))
        dataf =  Analyse_CV_coverletter('static/cv/' + secure_filename(f.filename), 'static/cover/' + secure_filename(f2.filename))
    return render_template('basic_tables.html',  tables=[dataf.to_html(classes='table table-bordered')], titles=dataf.columns.values)

@app.route('/basic_tables')
def basic_tables():
    return render_template('basic_tables.html')

@app.route('/basic_tables', methods=['GET', 'POST'])
def basic_tables2():
    if request.method == 'POST':
        f = request.files['file']
        f2 = request.files['file2']
        f.save('static/cv/' + secure_filename(f.filename))
        f2.save('static/cover/' + secure_filename(f2.filename))
        dataf =  Analyse_CV_coverletter('static/cv/' + secure_filename(f.filename), 'static/cover/' + secure_filename(f2.filename))
    return render_template('basic_tables.html',  tables=[dataf.to_html(classes='data')], titles=dataf.columns.values)