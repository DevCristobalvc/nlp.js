from sentence_transformers import SentenceTransformer
import json

def generate_embeddings(text_data):
    # Cargar un modelo preentrenado
    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = {}

    for page_num, paragraphs in text_data.items():
        page_embeddings = []
        for paragraph in paragraphs:
            # Generar embedding para cada párrafo
            embedding = model.encode(paragraph).tolist()  # Convertimos a lista para JSON
            page_embeddings.append(embedding)
        embeddings[page_num] = page_embeddings

    return embeddings

def save_embeddings(embeddings, output_path):
    with open(output_path, 'w') as f:
        json.dump(embeddings, f, indent=4)

if __name__ == "__main__":
    # Cargar el texto extraído del PDF
    with open('../data/pdf_text.json', 'r') as f:
        pdf_text = json.load(f)

    print('Datos de texto cargados:', pdf_text)

    # Generar los embeddings localmente
    embeddings = generate_embeddings(pdf_text)

    # Guardar los embeddings generados
    save_embeddings(embeddings, '../data/embeddings.json')
    print("Embeddings generados y guardados en 'data/embeddings.json'")
