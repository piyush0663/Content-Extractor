
from vectorstore import DB_urlingection
from retriver import Retriver

if __name__ == '__main__':
        
    urls = [
        
        "https://en.wikipedia.org/wiki/Machine_learning"
    ]

    vectore_store_instance = DB_urlingection()
    vector_store, sources = vectore_store_instance.ingest_urls(urls)
    retriver_ = Retriver(vector_store,sources)

   
    
    # Example question
    question = "What is Machine Learning" # enter your query
    answer = retriver_.answer_question(question)
    print("\nQuestion:", question)
    print("\nAnswer:", answer)