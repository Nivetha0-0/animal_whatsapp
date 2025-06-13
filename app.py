from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pymongo.mongo_client import MongoClient
from langchain_community.utils.math import cosine_similarity
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from enum import Enum
from pydantic import BaseModel, Field
import os
import dotenv

# Load .env variables
dotenv.load_dotenv()

app = Flask(__name__)

# Session memory
user_sessions = {}

# OpenAI & MongoDB setup
OPENAI_KEY = os.environ["OPENAI_KEY"]
MONGO_URI = os.environ["MONGODB_URI"]

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_KEY)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=OPENAI_KEY)
smaller_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_KEY)
larger_llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_KEY)

client = MongoClient(MONGO_URI)
db = client["pdf_file"]
collection = db["animal_bites"]

# Prompt for classification
tagging_prompt = ChatPromptTemplate.from_template("""
Extract the desired information from the following passage.
Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
""")

# Enums for structured classification
class QueryCategory(str, Enum):
    CASUAL = "Casual Greeting"
    SUBJECT = "Subject-Specific"

class RelevanceCategory(str, Enum):
    ANIMAL_BITE = "Animal Bite-Related"
    NOT_RELATED = "Not Animal Bite-Related"

class CasualSubject(BaseModel):
    description: QueryCategory = Field(
        description="Is the query a casual greeting or subject-specific?"
    )

class RelatedNot(BaseModel):
    description: RelevanceCategory = Field(
        description="Is the query related to animal bites or not?"
    )

@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    incoming_msg = request.values.get("Body", "").strip()
    user_number = request.values.get("From", "")

    if user_number not in user_sessions:
        user_sessions[user_number] = []
    chat_history = user_sessions[user_number]

    # 1. Convert message into a standalone input
    standalone_prompt = f"""
Rewrite the user's latest message to be self-contained.
If it's a casual message (e.g., 'thanks'), leave it unchanged.

Chat history: {chat_history}
Latest input: {incoming_msg}
"""
    modified_input = larger_llm.invoke(standalone_prompt).content
    modified_input = str(modified_input)

    # 2. Classify: Subject-Specific or Casual
    tag_input = tagging_prompt.invoke({"input": modified_input})
    tag_response = smaller_llm.invoke(tag_input).content
    tag_text = tag_response if isinstance(tag_response, str) else str(tag_response)

    category_obj = PydanticOutputParser(pydantic_object=CasualSubject).parse(tag_text)
    category = category_obj.description.value  # ✅ use .value from Enum

    # 3. Process Subject-Specific Query
    if category == "Subject-Specific":
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
            bot_response = llm.invoke(answer_prompt).content
        else:
            rel_tag_input = tagging_prompt.invoke({"input": modified_input})
            rel_tag_response = smaller_llm.invoke(rel_tag_input).content
            rel_text = rel_tag_response if isinstance(rel_tag_response, str) else str(rel_tag_response)

            rel_obj = PydanticOutputParser(pydantic_object=RelatedNot).parse(rel_text)
            relevance = rel_obj.description.value

            if relevance == "Not Animal Bite-Related":
                bot_response = (
                    "Sorry, I only specialize in questions related to animal bites. "
                    "Feel free to ask me anything about bites, symptoms, or treatment!"
                )
            else:
                bot_response = (
                    "I couldn’t find enough info to answer this now. Please try again later."
                )

    # 4. Casual greetings or messages
    else:
        casual_response = f"""You're a friendly assistant that answers messages about animal bites.
Respond to the following casually:
Message: {incoming_msg}"""
        bot_response = llm.invoke(casual_response).content

    # 5. Save conversation
    chat_history.append((incoming_msg, bot_response))

    # 6. Return WhatsApp message
    twilio_reply = MessagingResponse()
    twilio_reply.message(bot_response)
    return str(twilio_reply)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
