import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from utils import clean_data
from chains import Chain
from portfolio import Portfolio

def create_app(llm, portfolio, clean_data):

    #Create title for web page
    st.title('Cold Email Generatr')
    url_input = st.text_input("Enter URL")
    submit_button = st.button("Submit")

    if submit_button:
        try:
            loader = WebBaseLoader([url_input])
            page_data = loader.load().pop().page_content
            page_data = clean_data(page_data)
            portfolio.load_portfolio()
            jobs_res = llm.extract_job(page_data)

            for job in jobs_res:
                skills = job.get('skills', [])
                links = portfolio.query_links(skills)
                email = llm.write_email(jobs_res, links)
                st.code(email, language='markdown')

        except Exception as e:
            st.error(f'An error occurred {e}')    

if __name__=='__main__':
    llm = Chain() 
    portfolio = Portfolio()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_app(llm, portfolio, clean_data)  
