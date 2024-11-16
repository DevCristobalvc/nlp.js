const { NlpManager } = require('node-nlp');
const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
//const { cosineSimilarity } = require('./utils'); // Función de similitud coseno (más abajo)


const app = express();
app.use(cors());
const port = 3000;
const manager = new NlpManager({ languages: ['en'] });

app.use(bodyParser.json({ limit: '50mb' }));  // Aumentar el límite de tamaño (por ejemplo, 50 MB)


// Endpoint básico de "Hola Mundo"
app.get('/', (req, res) => {
    res.send('Hola Mundo');
});

// Endpoint para generar embeddings
app.post('/generate-embeddings', async (req, res) => {
    const textData = req.body;
    console.log('Texto recibido:', textData);  // Verifica que los datos llegan correctamente


    const embeddings = {};

    // Entrenamos NLP.js con el texto extraído
    for (const [pageNum, paragraphs] of Object.entries(textData)) {
        const pageEmbeddings = [];
        for (const paragraph of paragraphs) {
            const response = await manager.process('en', paragraph);
            pageEmbeddings.push(response.intent); // Guardamos el "intent" como embedding
        }
        embeddings[pageNum] = pageEmbeddings;
    }

    console.log('Embeddings generados:', embeddings);  // Verifica los embeddings generados
    res.json(embeddings); // Devuelve los embeddings generados
});

// Endpoint para responder a preguntas
app.post('/ask-question', async (req, res) => {
    const question = req.body.question;
    const embeddings = req.body.embeddings;
    
    // Procesar la pregunta con NLP.js
    const response = await manager.process('en', question);
    
    // Encontrar el embedding más relevante
    let bestMatch = null;
    let highestSimilarity = -1;

    // Recorremos los embeddings para encontrar la coincidencia más cercana
    for (const pageNum in embeddings) {
        const pageEmbeddings = embeddings[pageNum];
        
        pageEmbeddings.forEach(embedding => {
            const similarity = cosineSimilarity(response.intent, embedding); // Compara la pregunta con el embedding
            if (similarity > highestSimilarity) {
                highestSimilarity = similarity;
                bestMatch = { pageNum, embedding };
            }
        });
    }

    // Si encontramos una coincidencia, respondemos con la página y el embedding más similar
    if (bestMatch) {
        res.json({ answer: `La respuesta más relevante se encuentra en la página ${bestMatch.pageNum} con el embedding: ${bestMatch.embedding}` });
    } else {
        res.json({ answer: "No se encontró una respuesta relevante." });
    }
});

// Función para calcular la similitud coseno
function cosineSimilarity(str1, str2) {
    const vector1 = stringToVector(str1);
    const vector2 = stringToVector(str2);
    
    const dotProduct = vector1.reduce((sum, val, idx) => sum + val * vector2[idx], 0);
    const magnitude1 = Math.sqrt(vector1.reduce((sum, val) => sum + val * val, 0));
    const magnitude2 = Math.sqrt(vector2.reduce((sum, val) => sum + val * val, 0));
    
    return dotProduct / (magnitude1 * magnitude2);
}

// Función simple para convertir una cadena de texto en un vector de características (por ejemplo, usando el índice de caracteres)
function stringToVector(str) {
    const vector = Array(256).fill(0); // Tamaño fijo de vector
    for (let i = 0; i < str.length; i++) {
        vector[str.charCodeAt(i) % 256] += 1; // Incrementar el índice correspondiente en el vector
    }
    return vector;
}


(async () => {
    manager.addDocument('en', 'hello', 'greet'); // Ejemplo de entrenamiento
    manager.addDocument('en', 'how are you', 'greet');
    manager.addDocument('en', 'goodbye', 'farewell');

    await manager.train(); // Entrenar el modelo
    manager.save(); // Guardar el modelo entrenado

    // Iniciar el servidor
    app.listen(port, () => {
        console.log(`Servidor NLP.js corriendo en http://localhost:${port}`);
    });
})();
