# import
import os
from flask import Flask, request, jsonify, render_template
from pgvector.asyncpg import register_vector
import asyncio
import asyncpg
from google.cloud.sql.connector import Connector
from dotenv import load_dotenv
# gcloud sql instances describe pg15-embeddings-pgvector-demo --project=chatbot-with-rag-447503

min_age = 0
max_age = 100


# GCP project details
project_id = "chatbot-with-rag-447503" 
database_password = "19991028" 
region = "us-central1" 
instance_name = "pg15-embeddings-pgvector-demo"
database_name = "clothing"
database_user = "clothing-admin" 


app = Flask(__name__)

def embed_qe_api(user_query):

    assert user_query, "⚠️ Please input a valid input search text"

    from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

    print('encoding with api service')
    load_dotenv()
    inference_api_key = os.getenv("inference_api_key")
    embeddings_service = HuggingFaceInferenceAPIEmbeddings(
    api_key=inference_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
)

    qe = embeddings_service.embed_query(user_query)
    return qe

def embed_qe_local(user_query):

    assert user_query, "⚠️ Please input a valid input search text"

    from langchain_huggingface import HuggingFaceEmbeddings
    import torch
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    # "show_progress": True  # Enable progress bar
    }
    if torch.cuda.is_available():
        print('encoding with cuda')
    else:
        print('encoding with cpu')
    embeddings_service = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs)

    qe = embeddings_service.embed_query(user_query)
    return qe


async def search_embedding_space(qe):
    matches = []
    loop = asyncio.get_running_loop()
    async with Connector(loop=loop) as connector:
        # Create connection to Cloud SQL database.
        conn: asyncpg.Connection = await connector.connect_async(
            f"{project_id}:{region}:{instance_name}",  # Cloud SQL instance connection name
            "asyncpg",
            user=f"{database_user}",
            password=f"{database_password}",
            db=f"{database_name}",
        )

        await register_vector(conn)
        similarity_threshold = 0.5 # was 0.7 but too high
        num_matches = 3

        # Find similar products to the query using cosine similarity search
        # over all vector embeddings, then use index as handle to associate two tables
        results = await conn.fetch(
            """
                            WITH vector_matches AS (
                              SELECT index, 1 - (embedding <=> $1) AS similarity
                              FROM clothing_embeddings
                              WHERE 1 - (embedding <=> $1) > $2
                              ORDER BY similarity DESC
                              FETCH FIRST $3 ROWS ONLY
                            )
                            SELECT clothing_id, review, age FROM clothing
                            WHERE index IN (SELECT index FROM vector_matches)
                            AND age >= $4 AND age <= $5
                            """,
            qe,
            similarity_threshold,
            num_matches,
            min_age,
            max_age,
        )

        if len(results) == 0:
            # raise Exception("Did not find any results. Adjust the query parameters.")
            matches.append(
                """No clothing item was returned. """
            )
            available = False
        else:
            available = True
            for r in results:
                # Collect the description for all the matched similar items
                matches.append(
                    f"""The clothing ID is {r["clothing_id"]}.
                            The review is ${r["review"]}.
                            """
                )
        await conn.close()

    return available, matches

def LLM_get_response(user_query, matches):

    from langchain_google_vertexai import ChatVertexAI
    from langchain.chains.summarize import load_summarize_chain
    from langchain.docstore.document import Document
    from langchain_core.prompts import PromptTemplate

    llm = ChatVertexAI(
        model="gemini-1.0-pro",
        #temperature=0,
        project=project_id)

    map_prompt_template = """
                You will be given a piece of review text of a clothing product.
                This description is enclosed in triple backticks (```).
                Using this description only, extract the ID and the features of the clothing.

                ```{text}```
                SUMMARY:
                """
    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

    combine_prompt_template = """
                    You will be given about 3 descriptions of different clothings
                    enclosed in triple backticks (```) and a question enclosed in
                    double backticks(``).
                    For every clothing item that you received,
                    answer how the clothing match the question based its features.
                    Mention all clothing items that you received.
                    The answer should be in as much detail as possible.
                    You should only use the information in the description.
                    Your answer should include the ID of the clothings and their features.
                    Your answer should be less than 300 words.
                    Your answer should be in Markdown in a numbered list format.
                    Do not include a conclusion or a note.


                    Description:
                    ```{text}```


                    Question:
                    ``{user_query}``


                    Answer:
                    """
    combine_prompt = PromptTemplate(
        template=combine_prompt_template, input_variables=["text", "user_query"]
    )

    docs = [Document(page_content=t) for t in matches]
    chain = load_summarize_chain(
        llm, chain_type="map_reduce", map_prompt=map_prompt, combine_prompt=combine_prompt
    )

    response = chain.run(
        {
            "input_documents": docs,
            "user_query": user_query,
        }
    )

    return response


# Chatbot main function
async def chatbot_response(user_query):
    
    # qe = embed_qe_local(user_query)
    qe = embed_qe_api(user_query)
    print('query embeded')

    # seach in database
    available, matches = await search_embedding_space(qe)
    print('matches returned')
    if(available == False):
        return "Did not find any matching item"
    else:
        response = LLM_get_response(user_query, matches)
        return response



####### webpage stuff #######

# Route to serve the HTML form
@app.route('/')
def home():
    return render_template('index.html')


# Create an endpoint to receive chatbot messages
@app.route('/chatbot', methods=['POST'])
async def chatbot():
    data = request.get_json()
    user_input = data.get('message', '')
    
    # Get chatbot response
    # async functions returns a coroutine
    # await = async function returns what the async function needs to return
    response = await chatbot_response(user_input)
    
    return jsonify({"response": response})

# Run the Flask app
if __name__ == '__main__':
        app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))