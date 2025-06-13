from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pymongo.mongo_client import MongoClient
from langchain_community.utils.math import cosine_similarity
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from enum import Enum
from pydantic import BaseModel, Field
import os
import dotenv

# Load environment variables
dotenv.load_dotenv()

app = Flask(__name__)

# Session memory to store chat history per user
user_sessions = {}

# Setup OpenAI and MongoDB credentials
OPENAI_KEY = os.environ["OPENAI_KEY"]
MONGO_URI = os.environ["MONGODB_URI"]

# Initialize embeddings and language models
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_KEY)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=OPENAI_KEY)
smaller_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_KEY)
larger_llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_KEY)

# MongoDB client and collection
client = MongoClient(MONGO_URI)
db = client["pdf_file"]
collection = db["animal_bites"]

# Prompt template for classification
tagging_prompt = ChatPromptTemplate.from_template("""
Extract the desired information from the following passage.
Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
""")

# Enum for query categories
class QueryCategory(str, Enum):
    CASUAL = "Casual Greeting"
    SUBJECT = "Subject-Specific"

# Enum for relevance
class RelevanceCategory(str, Enum):
    ANIMAL_BITE = "Animal Bite-Related"
    NOT_RELATED = "Not Animal Bite-Related"

# Pydantic model for casual vs subject-specific classification
class CasualSubject(BaseModel):
    description: QueryCategory = Field(description="Is the query a casual greeting or subject-specific?")

# Pydantic model for relevance classification
class RelatedNot(BaseModel):
    description: RelevanceCategory = Field(description="Is the query related to animal bites or not?")

@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    incoming_msg = request.values.get("Body", "").strip()
    user_number = request.values.get("From", "")

    # Initialize user chat history if new user
    if user_number not in user_sessions:
        user_sessions[user_number] = []
    chat_history = user_sessions[user_number]

    # Step 1: Rewrite user message to standalone input
    standalone_prompt = f"""
Rewrite the user's latest message to be self-contained.
If it's a casual message (e.g., 'thanks'), leave it unchanged.

Chat history: {chat_history}
Latest input: {incoming_msg}
"""
    modified_input_response = larger_llm.invoke(standalone_prompt)
    modified_input = getattr(modified_input_response, "content", str(modified_input_response)).strip()

    # Step 2: Classify message as Casual or Subject-Specific using Pydantic parser
    parser = PydanticOutputParser(pydantic_object=CasualSubject)
    format_instructions = parser.get_format_instructions()

    classification_prompt = f"""
Extract the desired information from the following passage.
Only extract the properties mentioned in the 'Classification' function.
Respond in JSON format.

Passage:
{modified_input}

{format_instructions}
"""
    classification_response = smaller_llm.invoke(classification_prompt)
    classification_text = getattr(classification_response, "content", str(classification_response)).strip()

    try:
        category_obj = parser.parse(classification_text)
        category = category_obj.description
    except Exception as e:
        print("Classification parsing failed:", e)
        category = QueryCategory.CASUAL  # fallback to casual to avoid crash

    # Step 3: Process subject-specific queries
    if category == QueryCategory.SUBJECT:
        embedding = embeddings_model.embed_query(modified_input)

        results = collection.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embeddings",
                    "queryVector": embedding,
                    "numCandidates": 100,
                    "limit": 3
                }
            }
        ])

        context = ""
        for doc in results:
            score = cosine_similarity([doc["embeddings"]], [embedding])[0][0]
            if score >= 0.44:
                context += doc["raw_data"] + "\n\n"

        if context:
            answer_prompt = f"""You are a helpful assistant that answers questions about animal bites.
Use this context to answer the question:
Context: {context}
Question: {modified_input}"""
            bot_response_obj = llm.invoke(answer_prompt)
            bot_response = getattr(bot_response_obj, "content", str(bot_response_obj))
        else:
            # Check relevance if no context found
            relevance_parser = PydanticOutputParser(pydantic_object=RelatedNot)
            rel_format_instructions = relevance_parser.get_format_instructions()

            relevance_prompt = f"""
Extract the desired information from the following passage.
Only extract the properties mentioned in the 'Classification' function.
Respond in JSON format.

Passage:
{modified_input}

{rel_format_instructions}
"""
            relevance_response = smaller_llm.invoke(relevance_prompt)
            relevance_text = getattr(relevance_response, "content", str(relevance_response)).strip()

            try:
                rel_obj = relevance_parser.parse(relevance_text)
                relevance = rel_obj.description
            except Exception as e:
                print("Relevance parsing failed:", e)
                relevance = RelevanceCategory.NOT_RELATED  # fallback

            if relevance == RelevanceCategory.NOT_RELATED:
                bot_response = (
                    "Sorry, I only specialize in questions related to animal bites. "
                    "Feel free to ask me anything about bites, symptoms, or treatment!"
                )
            else:
                bot_response = (
                    "I couldn’t find enough info to answer this now. Please try again later."
                )
    else:
        # Step 4: Casual greetings or messages
        casual_prompt = f"""You're a friendly assistant that answers messages about animal bites.
Respond to the following casually:
Message: {incoming_msg}"""
        casual_response_obj = llm.invoke(casual_prompt)
        bot_response = getattr(casual_response_obj, "content", str(casual_response_obj))

    # Step 5: Save conversation to session memory
    chat_history.append((incoming_msg, bot_response))

    # Step 6: Return Twilio WhatsApp message
    twilio_reply = MessagingResponse()
    twilio_reply.message(bot_response)
    return str(twilio_reply)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
