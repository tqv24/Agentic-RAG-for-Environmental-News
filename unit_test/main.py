import os
import time
import logging
import torch
import pandas as pd
import streamlit as st
from typing import List, Dict
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


# Configure page
st.set_page_config(
    page_title="Multiple LLM models evaluation with Environmental News",
    page_icon="🤖",
    layout="wide"
)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Set device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
# Main title
st.title("Multiple LLM models evaluation with Environmental News")
st.markdown("""
This system uses multiple AI agents and a LoRA fine-tuned model to answer your questions about environmental news.
""")


# Load LoRA fine-tuned model
def load_lora_model(model_path="./lora_model"):
    try:
        config = PeftConfig.from_pretrained(model_path)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
        lora_model = PeftModel.from_pretrained(base_model, model_path)
        merged_model = lora_model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        merged_model.to(device)
        return merged_model, tokenizer
    except Exception as e:
        st.error(f"Error loading LoRA model: {str(e)}")
        # Fallback to base model if LoRA model fails to load
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn").to(device)
        return model, tokenizer
def initialize_components():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    
    vectorstore = PineconeVectorStore(
        index_name="env-news",
        embedding=embeddings,
        pinecone_api_key=pinecone_api_key,
        text_key="raw_text_index"
    )
    
    return llm, embeddings, vectorstore

def load_chroma_db():
    df = pd.read_csv('news.csv')
    df = df[['Title', 'Authors', 'Article Text', 'Date Published']].dropna()
    docs = []
    for _, row in df.iterrows():
        metadata = {
            "title": row["Title"],
            "authors": row["Authors"],
            "date_published": row["Date Published"]
        }
        docs.append(Document(page_content=row["Article Text"], metadata=metadata))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    doc_splits = text_splitter.split_documents(docs)

    vectorstore2 = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings(model="text-embedding-3-small")
    )
    return vectorstore2

# Initialize components if API keys are provided
if openai_api_key and pinecone_api_key and not st.session_state.initialized:
    with st.spinner("Initializing components..."):
        # Load components
        llm, embeddings, vectorstore = initialize_components()
        vectorstore2 = load_chroma_db()
        retriever = vectorstore2.as_retriever(search_kwargs={"k": 5})
        lora_model, lora_tokenizer = load_lora_model()
        st.session_state.initialized = True
        st.success("Components initialized successfully!")
elif st.session_state.initialized:
    llm, embeddings, vectorstore = initialize_components()
    vectorstore2 = load_chroma_db()
    retriever = vectorstore2.as_retriever(search_kwargs={"k": 5})
    lora_model, lora_tokenizer = load_lora_model()

# Define utility functions
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def process_documents(docs):
    processed_docs = []
    for doc in docs:
        # Clean and format the content
        content = doc.page_content.strip()
        
        # Enhanced metadata
        metadata = {
            'raw_text_index': str(len(processed_docs)),
            'title': doc.metadata.get('title', ''),
            'authors': doc.metadata.get('authors', ''),
            'date_published': str(doc.metadata.get('date_published', '')),
            'summary': content[:200] + "..."
        }
        
        new_doc = Document(
            page_content=content,
            metadata=metadata
        )
        processed_docs.append(new_doc)
    return processed_docs

# Function to generate summaries using LoRA model
def generate_summary(model, tokenizer, article_text, max_length=130, min_length=30):
    input_text = "Summarize: " + article_text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            num_beams=3,
            early_stopping=True
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Define data models for grading
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

if st.session_state.initialized:
    # Create structured LLM graders
    structured_doc_grader = llm.with_structured_output(GradeDocuments, method="function_calling")
    structured_hallucination_grader = llm.with_structured_output(GradeHallucinations, method="function_calling")
    structured_answer_grader = llm.with_structured_output(GradeAnswer, method="function_calling")

    # Define prompts
    doc_grade_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a grader assessing relevance of a retrieved document to a user question. 
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])

    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ])

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a grader assessing whether an answer addresses / resolves a question.
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ])

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a question re-writer that converts an input question to a better version that is optimized
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ])

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."""),
        ("human", "Question: {question}\nContext: {context}"),
    ])

    summary_integration_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant that integrates document summaries to provide comprehensive answers.
        Use the following summaries to answer the question. Focus on being accurate and concise."""),
        ("human", "Question: {question}\nSummaries: {summaries}"),
    ])

    # Create chains
    retrieval_grader = doc_grade_prompt | structured_doc_grader
    hallucination_grader = hallucination_prompt | structured_hallucination_grader
    answer_grader = answer_prompt | structured_answer_grader
    question_rewriter = rewrite_prompt | llm | StrOutputParser()
    rag_chain = rag_prompt | llm | StrOutputParser()
    summary_integration_chain = summary_integration_prompt | llm | StrOutputParser()

    # Define graph state
    class GraphState(TypedDict):
        """Represents the state of our graph."""
        question: str
        original_question: str
        generation: str
        documents: List
        summaries: List
        cycle_count: int

    # Define nodes (agents)
    def retrieve_documents(state: Dict) -> Dict:
        """Retrieves documents based on the query."""
        question = state["question"]
        original_question = state.get("original_question", question)
        cycle_count = state.get("cycle_count", 0)
        
        # Retrieve documents
        documents = retriever.get_relevant_documents(question)
        
        return {
            "documents": documents, 
            "question": question,
            "original_question": original_question,
            "cycle_count": cycle_count,
            "generation": state.get("generation", ""),
            "summaries": state.get("summaries", [])
        }

    def grade_documents(state: Dict) -> Dict:
        """Grades documents for relevance to the question."""
        question = state["question"]
        original_question = state["original_question"]
        documents = state["documents"]
        cycle_count = state.get("cycle_count", 0)
        
        # Score each doc
        filtered_docs = []
        for d in documents:
            # If we've cycled too many times, consider all docs relevant
            if cycle_count >= 3:
                filtered_docs = documents
                break
                
            score = retrieval_grader.invoke(
                {"question": original_question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade.lower() == "yes":
                filtered_docs.append(d)
        
        return {
            "documents": filtered_docs, 
            "question": question,
            "original_question": original_question,
            "cycle_count": cycle_count,
            "generation": state.get("generation", ""),
            "summaries": state.get("summaries", [])
        }

    def generate_summaries(state: Dict) -> Dict:
        """Generates summaries using LoRA fine-tuned model."""
        question = state["question"]
        original_question = state["original_question"]
        documents = state["documents"]
        cycle_count = state.get("cycle_count", 0)
        
        summaries = []
        for doc in documents:
            summary = generate_summary(lora_model, lora_tokenizer, doc.page_content)
            summaries.append(summary)
        
        return {
            "documents": documents, 
            "question": question,
            "original_question": original_question,
            "summaries": summaries,
            "cycle_count": cycle_count,
            "generation": state.get("generation", "")
        }

    def generate_final_answer(state: Dict) -> Dict:
        """Generates final answer based on summaries and documents."""
        question = state["question"]
        original_question = state["original_question"]
        documents = state["documents"]
        summaries = state["summaries"]
        cycle_count = state.get("cycle_count", 0)
        
        # Format summaries and documents
        summaries_text = "\n\n".join(summaries)
        
        # Generate answer using summaries
        generation = summary_integration_chain.invoke({
            "summaries": summaries_text, 
            "question": original_question
        })
        
        return {
            "documents": documents, 
            "summaries": summaries, 
            "question": question,
            "original_question": original_question,
            "generation": generation,
            "cycle_count": cycle_count
        }

    def transform_query(state: Dict) -> Dict:
        """Transforms the query to produce a better question."""
        question = state["question"]
        original_question = state["original_question"]
        documents = state["documents"]
        cycle_count = state.get("cycle_count", 0) + 1
        
        # If we've cycled too many times, force an answer
        if cycle_count >= 5:
            # Generate a response even with limited relevance
            context = "\n\n".join([doc.page_content for doc in documents[:2]]) if documents else ""
            generation = f"I've searched for information about '{original_question}', but couldn't find highly relevant documents. Based on the available information, I can provide a limited response."
            
            return {
                "documents": documents, 
                "question": question,
                "original_question": original_question,
                "generation": generation,
                "cycle_count": cycle_count,
                "summaries": state.get("summaries", [])
            }
        
        # Re-write question
        better_question = question_rewriter.invoke({"question": original_question})
        
        return {
            "documents": documents, 
            "question": better_question,
            "original_question": original_question,
            "cycle_count": cycle_count,
            "generation": state.get("generation", ""),
            "summaries": state.get("summaries", [])
        }

    def force_generate(state: Dict) -> Dict:
        """Generate an answer even with limited document relevance."""
        question = state["question"]
        original_question = state["original_question"]
        cycle_count = state.get("cycle_count", 0)
        
        # Get fresh documents
        documents = retriever.get_relevant_documents(question)
        
        # Generate summaries
        summaries = []
        for doc in documents:
            summary = generate_summary(lora_model, lora_tokenizer, doc.page_content)
            summaries.append(summary)
        
        # Use whatever documents we have
        context = "\n\n".join([doc.page_content for doc in documents[:3]]) if documents else ""
        summaries_text = "\n\n".join(summaries)
        
        generation = summary_integration_chain.invoke({
            "summaries": summaries_text, 
            "question": original_question
        })
        
        return {
            "documents": documents, 
            "question": question,
            "original_question": original_question,
            "generation": generation,
            "cycle_count": cycle_count,
            "summaries": summaries
        }

    # Define edge functions
    def decide_to_generate(state: Dict) -> str:
        """Determine whether to generate an answer or transform the query."""
        filtered_documents = state["documents"]
        cycle_count = state.get("cycle_count", 0)
        
        # Force generation after several cycles
        if cycle_count >= 3:
            return "force_generate"
        elif not filtered_documents:
            return "not_relevant"
        else:
            return "relevant"

    def grade_generation(state: Dict) -> str:
        """Determine if the generation is grounded and answers the question."""
        question = state["original_question"]
        documents = state["documents"]
        generation = state["generation"]
        
        # Format documents for grading
        docs_text = "\n\n".join([doc.page_content for doc in documents])

        # Check hallucination
        score = hallucination_grader.invoke(
            {"documents": docs_text, "generation": generation}
        )
        grade = score.binary_score
        
        if grade.lower() == "yes":
            # Check question-answering
            score = answer_grader.invoke({"question": question, "generation": generation})
            grade = score.binary_score
            if grade.lower() == "yes":
                return "useful"
            else:
                return "not_useful"
        else:
            return "not_supported"

    # Build the graph
    @st.cache_resource
    def build_workflow():
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("retrieve", retrieve_documents)
        workflow.add_node("grade_documents", grade_documents)
        workflow.add_node("generate_summaries", generate_summaries)
        workflow.add_node("generate_answer", generate_final_answer)
        workflow.add_node("transform_query", transform_query)
        workflow.add_node("force_generate", force_generate)
        
        # Add edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        
        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "not_relevant": "transform_query",
                "relevant": "generate_summaries",
                "force_generate": "force_generate"
            }
        )
        
        workflow.add_edge("generate_summaries", "generate_answer")
        workflow.add_edge("force_generate", END)
        workflow.add_edge("transform_query", "retrieve")
        
        workflow.add_conditional_edges(
            "generate_answer",
            grade_generation,
            {
                "not_supported": "generate_answer",
                "useful": END,
                "not_useful": "transform_query"
            }
        )
        
        # Compile
        return workflow.compile()
    def generate_summaries_with_base_llm(state: Dict) -> Dict:
        """Generates summaries using base LLM instead of LoRA model."""
        question = state["question"]
        original_question = state["original_question"]
        documents = state["documents"]
        cycle_count = state.get("cycle_count", 0)

        summaries = []
        for doc in documents:
            summary_prompt = f"Summarize the following text in a concise manner: {doc.page_content}"
            base_summary = llm.invoke(summary_prompt).content
            summaries.append(base_summary)

        return {
            "documents": documents, 
            "question": question,
            "original_question": original_question,
            "summaries": summaries,
            "cycle_count": cycle_count,
            "generation": state.get("generation", "")
        }

    # Create a modified workflow with base LLM for summarization
    base_workflow = StateGraph(GraphState)

    # Add nodes (same as original workflow but with base LLM summarization)
    base_workflow.add_node("retrieve", retrieve_documents)
    base_workflow.add_node("grade_documents", grade_documents)
    base_workflow.add_node("generate_summaries", generate_summaries_with_base_llm)  # Use base LLM version
    base_workflow.add_node("generate_answer", generate_final_answer)
    base_workflow.add_node("transform_query", transform_query)
    base_workflow.add_node("force_generate", force_generate)

    # Add edges (same as original workflow)
    base_workflow.add_edge(START, "retrieve")
    base_workflow.add_edge("retrieve", "grade_documents")

    base_workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "not_relevant": "transform_query",
            "relevant": "generate_summaries",
            "force_generate": "force_generate"
        }
    )

    base_workflow.add_edge("generate_summaries", "generate_answer")
    base_workflow.add_edge("force_generate", END)
    base_workflow.add_edge("transform_query", "retrieve")

    base_workflow.add_conditional_edges(
        "generate_answer",
        grade_generation,
        {
            "not_supported": "generate_answer",
            "useful": END,
            "not_useful": "transform_query"
        }
    )

    # Compile the base workflow
    base_app = base_workflow.compile()


    # Create basic RAG chain for comparison
    def create_basic_rag_chain():
        retriever_rag = vectorstore.as_retriever(search_kwargs={"k": 5})
        prompt = hub.pull("rlm/rag-prompt")
        
        chain = (
            {"context": retriever_rag | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

    # Evaluation function
    def evaluate_system(question):
        # Base LLM (no RAG)
        base_llm_response = llm.invoke(question).content

        # Basic RAG
        basic_rag_chain = create_basic_rag_chain()
        basic_rag_response = basic_rag_chain.invoke(question)

        # Advanced RAG with base model (no fine-tuning)
        base_inputs = {"question": question, "original_question": question, "cycle_count": 0}
        base_result = base_app.invoke(base_inputs)
        advanced_rag_base_response = base_result["generation"]

        # Advanced RAG with LoRA model
        app = build_workflow()
        inputs = {"question": question, "original_question": question, "cycle_count": 0}
        result = app.invoke(inputs)
        advanced_rag_lora_response = result["generation"]

        return {
            "base_llm": base_llm_response,
            "basic_rag": basic_rag_response,
            "advanced_rag_base": advanced_rag_base_response,
            "advanced_rag_lora": advanced_rag_lora_response
        }


# # Allow user to enter their own question
# eval_question = st.text_input("Enter your question:")

# if st.button("Generate Answer") and st.session_state.initialized:
#     if eval_question:
#         with st.spinner("Evaluating system configurations... This may take a minute."):
#             results = evaluate_system(eval_question)
            
#             # Display results
#             st.subheader("Evaluation Results")
            
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.markdown("**Base LLM (no RAG)**")
#                 st.write(results["base_llm"])
                
#                 st.markdown("**Advanced RAG with Base Model**")
#                 st.write(results["advanced_rag_base"])
            
#             with col2:
#                 st.markdown("**Basic RAG**")
#                 st.write(results["basic_rag"])
                
#                 st.markdown("**Advanced RAG with LoRA Model**")
#                 st.write(results["advanced_rag_lora"])
#     else:
#         st.warning("Please enter a question.")
# elif not st.session_state.initialized and st.button("Generate Answer"):
#     st.error("Please provide API keys to initialize the system.")    
    
# =============================
# 🌐 Streamlit Interface Layout
# =============================


st.markdown("---")
st.header("🧠 System Evaluation Interface")

# Section 1: Input
st.subheader("1. Question Input")
eval_question = st.text_input("🔍 Enter your question about environmental news")

# Section 2: Run Evaluation
st.subheader("2. Run Evaluation")

if st.button("🚀 Run Evaluation") and st.session_state.initialized:
    if eval_question:
        with st.spinner("Evaluating all system configurations..."):
            results = evaluate_system(eval_question)

            # Section 3: Display Results
            st.markdown("---")
            st.subheader("3. 🔎 Evaluation Results")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🤖 Base LLM (no RAG)")
                st.info(results["base_llm"])
                
                st.markdown("### 📚 Basic RAG")
                st.info(results["basic_rag"])


            with col2:

                st.markdown("### 🔍 Advanced RAG (Base Model)")
                st.success(results["advanced_rag_base"])

                st.markdown("### 🚀 Advanced RAG (LoRA Model)")
                st.success(results["advanced_rag_lora"])
    else:
        st.warning("⚠️ Please enter a question to proceed.")

elif not st.session_state.initialized and st.button("🚀 Run Evaluation"):
    st.error("❌ Please ensure API keys are configured to initialize the system.")
