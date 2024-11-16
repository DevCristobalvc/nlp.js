from flask import Flask, request, jsonify
import json
import requests

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    query_text = request.json['query']
    
    # Generar el embedding de la consulta a través del servidor NLP.js
    response = requests.post('http://localhost:3000/generate-embeddings', json={'query': [query_text]})
    query_embedding = response.json()
    
    # Cargar los embeddings generados previamente desde el archivo
    with open('data/embeddings.json', 'r') as f:
        embeddings = json.load(f)
    
    # Buscar los párrafos más relevantes
    # Aquí puedes aplicar alguna lógica de similitud, como comparar los "intents"
    similarities = []
    for page_num, page_embeddings in embeddings.items():
        for para_num, embedding in enumerate(page_embeddings):
            similarity = 1 if query_embedding['query'][0] == embedding else 0
            similarities.append((similarity, page_num, para_num))
    
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_results = similarities[:2]

    # Devolver los párrafos más relevantes
    results = []
    with open('data/pdf_text.json', 'r') as f:
        pdf_text = json.load(f)
        for _, page_num, para_num in top_results:
            results.append(pdf_text[str(page_num)][para_num])
    
    return jsonify({'results': results})

if __name__ == "__main__":
    app.run(debug=True)
