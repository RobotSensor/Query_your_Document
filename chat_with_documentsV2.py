import streamlit as st
#from pinecone import Pinecone, ServerlessSpec
#from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from pinecone import ServerlessSpec
#from pinecone.config import ServerlessSpec
from langchain.chat_models import ChatOpenAI
#from langchain_pinecome import PineconeVectorStore
from langchain_community.vectorstores import Pinecone
#from pinecone import Pinecone, ServerlessSpec
import os
#from openai import OpenAI




# Define a function to load the file

def load_document(files):
    import os
    name, extension = os.path.splitext(files)
    
    # for uploading PDF 
    if extension =='.pdf':
        from langchain.document_loaders import PyPDFLoader
        print (f'Loading {files}')
        loader = PyPDFLoader(files)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Load {files}')
        loader = Docx2txtLoader(files)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(files)

    else:
        print('Document format is not supported!')
        return None
        
    data = loader.load()
    
    return data



# Define a function to chunk the data

# define a function to chunk the data into pieces
def chunk_data(data, chunk_size = 256, chunk_overlap=0):
    
    from langchain.text_splitter import RecursiveCharacterTextSplitter # use for chunk the data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks


# Create a function to delete all existing indices

# Delete existing index from Pinecone index
def delete_pinecone_index(index_name ='all'):
    
    import pinecone
    from langchain_pinecone import Pinecone, PineconeVectorStore
    pc = pinecone.Pinecone() 
    if index_name == 'all':
        indexes = pc.list_indexes().names()
        print('Deleting all indexes....')

        for index in indexes:
            pc.delete_index(index)
        print('Ok')
    else: 
        print(f'Deleting index {index_name} ...', end='')
        pc.delete_index(index_name)
        print('Ok')


 # Create a function to create a new Pinecone index
 
def create_pinecone_index(index_name, dimension=384, metric='cosine'):
    import pinecone
    from pinecone import ServerlessSpec
    from langchain_pinecone import Pinecone, PineconeVectorStore
    pc = pinecone.Pinecone()
    if index_name not in pc.list_indexes().names():
        print(f'Creating index {index_name}...')
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        
        )
    print('Done' )      


# Create a function to insert the Embedding into the Pinecone Index
def insert_embeddings(chunks, index_name):
    
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain_pinecone import Pinecone, PineconeVectorStore

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = PineconeVectorStore.from_texts(
        texts=[c.page_content for c in chunks],
        embedding=embeddings,
        index_name=index_name
    )
    return vector_store


# Asking Questions (Similarity Search) with OpenAI

# create a function for asking a question and answering
def question_answer( vector_store, query, k=3):
    
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-5-nano', temperature=1) # gpt-4.1-nano #'gpt-5-nano'
    retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs={'k':k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.run(query)
    return answer

# Asking Questions (Similarity Search) without OpenAI

def question_answer_simple(vector_store, q, k=3):   #k=3
    results = vector_store.similarity_search(q, k=k)
    for res in results:
        response = res.page_content
    return response




def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']



# Application entry point 

if __name__ == "__main__":

    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override =True)
    

    
    with st.columns(3)[1]:
     st.image('im7.png')
     #st.image('im4.png', width=100)   
     #st.subheader('Application⚙️')
     #st.image('im3.png', width=80)
     
    with st.columns(3)[0]:
     st.image('im3.png', width=60)
     #st.image('im4.png', width=100)   
     st.write('Application⚙️')
     #st.image('im3.png', width=80)
        
 
    with st.sidebar:
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        
        Uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)
        add_data = st.button('Add data', on_click=clear_history)


        if Uploaded_file and add_data:
            with st.spinner('Reading, chunking and embedding file ...'):
                bytes_data = Uploaded_file.read()
                file_name = os.path.join('./', Uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)


                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunks size: {chunk_size}, Chunks: {len(chunks)}')

                # Call function to delete Pinecone index 
                delete_pinecone_index()

                # create new pinecone index
                index_name ='askdocument'
                create_pinecone_index(index_name, dimension=384, metric='cosine')

                vector_store = insert_embeddings(chunks, index_name)

                #while True:
                #question_answer_simple(vector_store, q, k)

                st.session_state.vs = vector_store
                #question_answer_simple(vector_store, q, k=3)
                
                st.success('File Uploaded, chunked and embadded successfully.')

    st.markdown("<hr style='border: 2px solid #DC143C;'>", unsafe_allow_html=True)
    
    st.write('🔍 With OPENAI')
    query = st.text_input(' 👇 ')
    if query:

        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            #st.write(f'k: {k}')
            answer = question_answer( vector_store, query, k)
            st.text_area('LLM Answer  : ', value=answer)


    #st.divider()
    st.markdown("<hr style='border: 2px solid #4CAF50;'>", unsafe_allow_html=True)
    
    st.write('🔍 Without OPENAI')
    q = st.text_input('👇 ')    
    if q:
        if 'vs' in st.session_state:
             vector_store = st.session_state.vs
             st.write(f'k: {k}')
             response = question_answer_simple(vector_store, q, k)
             st.text_area('LLM response  : ', value=response)





    #st.divider()
    st.markdown("<hr style='border: 2px solid #DC143C;'>", unsafe_allow_html=True)
    if 'history' not in st.session_state:

        st.session_state.history = ''
    value = f'Q: {query} \nA: {answer}'
    st.session_state.history = f' {value} \n {"-" * 100} \n {st.session_state.history}'
    h = st.session_state.history
    st.text_area(label='Chat History', value=h, key='history', height=400)

