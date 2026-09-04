from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

ai_researchPapers=Path(__file__).resolve().parent.parent.parent / 'data/raw/papers'

def load_documents():
     docs=[]
     for x in ai_researchPapers.iterdir():
         loader=PyPDFLoader(
              file_path=x,
         )
         docs.extend(loader.load())
     return docs
        

