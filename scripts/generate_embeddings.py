import requests
import json

def generate_embeddings(text_data):
    url = 'http://localhost:3000/generate-embeddings'
    response = requests.post(url, json=text_data)
    
    # Verifica el código de estado HTTP
    if response.status_code != 200:
        print(f"Error al hacer la solicitud: {response.status_code}")
        print(response.text)  # Ver los detalles de la respuesta
        return {}

    try:
        embeddings = response.json()
    except ValueError as e:
        print(f"Error al parsear JSON: {e}")
        print(response.text)  # Ver detalles de la respuesta
        return {}

    return embeddings

def ask_question(question, embeddings):
    url = 'http://localhost:3000/ask-question'
    data = {'question': question, 'embeddings': embeddings}
    response = requests.post(url, json=data)
    answer = response.json()
    return answer

def save_embeddings(embeddings, output_path):
    with open(output_path, 'w') as f:
        json.dump(embeddings, f, indent=4)

if __name__ == "__main__":
    # Cargar el texto extraído del PDF
    with open('../data/pdf_text.json', 'r') as f:  # Ajustar para subir un nivel
        pdf_text = json.load(f)

    print('Datos de texto cargados:', pdf_text)  # Verifica que los datos se estén cargando correctamente

    # Generar los embeddings
    embeddings = generate_embeddings(pdf_text)

    # Guardar los embeddings generados
    save_embeddings(embeddings, '../data/embeddings.json')
    print(f"Embeddings generados y guardados en data/embeddings.json")

    # Hacer una pregunta al sistema
    question = "What is the software engineering body of knowledge?"
    answer = ask_question(question, embeddings)
    print(f"Respuesta a la pregunta: {answer['answer']}")
