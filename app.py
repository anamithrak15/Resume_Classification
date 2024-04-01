import nltk
import re
import pickle
import streamlit as st #Streamlit is an open-source Python library that allows you to create interactive web applications for machine learning, data science, and other data-driven tasks with minimal effort.
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
nltk.download('punkt')
nltk.download('stopwords')

#loading model
clf=pickle.load(open('clf.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

#Web
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def clean_resume(text):
    cleantxt=re.sub('http\S+\s',' ',text)
    cleantxt=re.sub('@\S+',' ',cleantxt)
    cleantxt=re.sub('#\S+\s',' ', cleantxt)
    cleantxt=re.sub('RT|cc',' ',cleantxt)
    cleantxt=re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""),' ',cleantxt)
    cleantxt=re.sub(r'[^\x00-\x7f]',' ',cleantxt)
    cleantxt=re.sub('\s+',' ',cleantxt)
    return cleantxt

def main():
    st.title("Resume Screening App")
    st.write("Upload your resume to see the predicted category")
    
    #upload resume
    upload_file = st.file_uploader("Upload Your Resume", type=["txt", "pdf"])
    
    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            #if utf-8 fails try latin-1
            resume_text = resume_bytes.decode('latin-1')
        
        cleaned_resume = clean_resume(resume_text)
        input_features = tfidf.transform([cleaned_resume])
        pred_id = clf.predict(input_features)[0]
        # st.write(pred_id)
        
        
        category_mapping = {
    6: 'Data Science',
    12: 'HR',
    0: 'Advocate',
    1: 'Arts',
    24: 'Web Designing',
    16: 'Mechanical Engineer',
    22: 'Sales', 
    14: 'Health and fitness', 
    5: 'Civil Engineer',
    15: 'Java Developer',
    4: 'Business Analyst',
    21: 'SAP Developer',
    2: 'Automation Testing',
    11: 'Electrical Engineering',
    18: 'Operations Manager',
    20: 'Python Developer',
    8: 'DevOps Engineer',
    17: 'Network Security Engineer',
    19:  'PMO', 
    7:  'Database',
    13: 'Hadoop',
    10: 'ETL Developer',
    9: 'DotNet Developer',
    3: 'Blockchain', 
    23:'Testing'
    }
        
        category_name = category_mapping.get(pred_id, "Unknown")
        st.write("Predicted Category to which your resume belongs to :  ", category_name)
    
#python main
if __name__ == "__main__":
    main()
    

    
