import os, re
import time
from flask import Flask, request, jsonify
import openai 
from dotenv import load_dotenv
import json
import logging
from utils import process_multiline_string, extract_documents_based_on_distance, \
    make_json_objects, filter_unique_parent_codes, prepare_retriever
from flask import Flask, request, jsonify



app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Configure OpenAI API Key
openai_key = os.getenv('OPENAI_API_KEY')
if not openai_key:
    logger.error("OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable.")
    raise ValueError("OpenAI API key not found.")


client = openai.OpenAI(
    api_key=openai_key,
)

#prepare retriever
retriever = prepare_retriever()

@app.route('/', methods=['GET'])
def home():
    logger.info("Health check accessed.")
    return jsonify({"status": "OK", "message": "Clinical Note Processor is running."}), 200


# Currently not in use, summarization triggered directly from the frontend
@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    if not data or 'clinical_note' not in data:
        return jsonify({"error": "Please provide 'clinical_note' in the JSON body."}), 400

    clinical_note = data['clinical_note']
    logger.info("Summarizing clinical note.")


    try:
        completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
                {
                    "role": "system",
                    "content": "You are a doctor's assist and you are great at summarizing clinical clinical notes. \
                    Try to keep information in about history of present illness, \
                    past medical history and physical exam. The identity of the patient is not important, it is enough to state \
                    the age and gender of the patient. Do not summarize too much, a good length is 300 words. \
                    You can start the summary directly, no need to explicitly state that this is a summary.",
                },
                {
                    "role": "user",
                    "content": clinical_note,
                }
            ]
        )
        summary = completion.choices[0].message.content
        return jsonify({"clinical_note_summary": summary}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/generate', methods=['POST'])
def generate():
    max_length = 150
    data = request.get_json()
    if not data or 'clinical_note_summary' not in data:
        return jsonify({"error": "Please provide 'clinical_note_summary' in the JSON body."}), 400

    clinical_note_summary = data['clinical_note_summary']
    logger.info("Model to generate ICD-10-CM codes.")
    clinical_note_summary = re.sub(r'\s+', ' ', clinical_note_summary).strip()

    # Prepare the prompt
    sys_prompt = f"""
You are an expert medical coding assistant.

Task: Analyze the following summary of a clinical note and provide a list of appropriate ICD-10-CM codes that best relate to the medical information mentioned.

Instructions:

-Provide a maximum of 4 ICD-10-CM codes.
-Format: [Code]: [Description]
-List each code and its description on a new line.
-Only include the codes and their descriptions—no extra text.

Clinical Note Summary:
"""
    
    logger.debug(f"sys_prompt: {sys_prompt}")
    logger.debug(f"clinical_note_summary: {clinical_note_summary}")
    if not isinstance(sys_prompt, str) or not isinstance(clinical_note_summary, str):
        raise ValueError("sys_prompt and clinical_note_summary must be strings.")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": sys_prompt,
                },
                {
                    "role": "user",
                    "content": clinical_note_summary,
                }
            ]
        )
        print(response)
    
        response_text = response.choices[0].message.content
        return jsonify({"response": response_text}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/rag', methods=['POST'])
def retrieve():
    docs = []

    data = request.get_json()
    if not data or 'query_text' not in data:
        return jsonify({"error": "Please provide 'query_text' in the JSON body."}), 400

    query_text = data['query_text']
    if not query_text:
        return jsonify({"error": "Provided 'query_text' cannot be empty."}), 400
    query_text = process_multiline_string(query_text)
    logger.info(f"query_text: {query_text}")

    for query in query_text:
        try:
            logger.info(f"Retrieving documents for query: {query}")
            response = retriever.retrieve(query)
            for ret in response:
                doc_text = ret.get_text()
                logger.info(f"doc_text: {doc_text}")
                docs.append(doc_text)
        except Exception as e:
            logger.error(f"Error retrieving documents for query: {query}")
            logger.error(f"Error: {e}")
            docs.append(f"Error retrieving documents for query: {query}")

    json_docs = make_json_objects(docs)
    logger.info("Filtering unique parent codes.")
    filtered_json_docs = filter_unique_parent_codes(json_docs)
    return jsonify({"rag_documents": filtered_json_docs}), 200



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
