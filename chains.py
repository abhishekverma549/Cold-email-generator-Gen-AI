from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
import os

from dotenv import load_dotenv
load_dotenv()

class Chain:
    def __init__(self):
        self.llm=ChatGroq(
                        model="llama-3.1-70b-versatile",
                        api_key = os.getenv("GROQ_API_KEY"),
                        temperature=0,
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                    )

    def extract_job(self, cleaned_text):

        prompt_extract = PromptTemplate.from_template("""
            ### Scraped text data from website:
            {page_data}
            The scraped text is from career's page of website.
            Your job is to extract the job position and returned in the jason format
            following keys: 'role', 'experience','skills' and 'description'.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
        """)    
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={'page_data': cleaned_text})

        try:
            json_parser=JsonOutputParser()
            json_res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Conent too big unable to parse job")
        
        return json_res if isinstance(json_res, list) else [json_res]
    
    def write_email(self, job, links):
        prompt_email = PromptTemplate.from_template("""
            ### JOB DESCRITION:
            {job_description}
            ### INSTRUCTION:
            You are XYZ, a business development executive at ABC. ABC is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools. 
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
            process optimization, cost reduction, and heightened overall efficiency. 
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of ABC 
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase Atliq's portfolio: {link_list}
            Remember you are XYZ, BDE at ABC. 
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
        """)
        chain_email = prompt_email|self.llm
        res = chain_email.invoke({'job_description':str(job), 'link_list':links})
        return res.content