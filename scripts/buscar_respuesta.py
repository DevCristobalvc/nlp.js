import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

from sentence_transformers import SentenceTransformer, util
import json

# Cargar el modelo preentrenado
model = SentenceTransformer('all-MiniLM-L6-v2')

# Cargar los embeddings precomputados y el contenido original del archivo JSON
with open('../data/embeddings.json', 'r') as f:
    embeddings_data = json.load(f)

with open('../data/pdf_text.json', 'r') as f:
    text_data = json.load(f)

# Función para buscar la respuesta más relevante
def buscar_respuesta(question):
    question_embedding = model.encode(question)

    best_match = None
    highest_similarity = -1
    best_page = None
    best_idx = None

    # Buscar el párrafo más relevante
    for page_num, page_embeddings in embeddings_data.items():
        for idx, paragraph_embedding in enumerate(page_embeddings):
            similarity = util.cos_sim(question_embedding, paragraph_embedding).item()
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = text_data[page_num][idx]  # Guardar el párrafo más relevante
                best_page = page_num
                best_idx = idx

    # Comprobar si se encontró una respuesta relevante
    if best_match is not None:
        if best_page is not None:
            pagina_para_imprimir = int(best_page) + 1

            print(f"\nLa respuesta más relevante se encuentra en la página {pagina_para_imprimir} con una similitud de {highest_similarity:.4f}")
            print("Contenido relevante:")

            # Imprimir el párrafo relevante y agregar párrafos antes y después para más contexto
            parrafos = text_data[best_page]
            contenido_ampliado = "\n\n".join(parrafos[max(0, best_idx - 1):min(len(parrafos), best_idx + 2)])
            print(contenido_ampliado)
        else:
            print("\nNo se encontró una respuesta relevante.")
    else:
        print("\nNo se encontró una respuesta relevante.")

# Menú de interacción con el usuario
def main():
    while True:
        print("\n--- Menú de Búsqueda de Respuestas ---")
        question = input("Escribe tu pregunta (o 'salir' para terminar): ")
        
        if question.lower() == 'salir':
            print("Gracias por usar el buscador de respuestas. ¡Adiós!")
            break
        
        buscar_respuesta(question)

# Ejecutar el programa principal
if __name__ == "__main__":
    main()
