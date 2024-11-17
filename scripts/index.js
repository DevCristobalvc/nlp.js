const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const fs = require('fs');
const natural = require('natural'); // Para manejo de similitud si necesitas una alternativa sencilla

const app = express();
app.use(cors());
const port = 3000;

app.use(bodyParser.json({ limit: '50mb' }));

// Cargar archivos JSON
const pdfText = JSON.parse(fs.readFileSync('../data/pdf_text.json', 'utf8'));
const embeddings = JSON.parse(fs.readFileSync('../data/embeddings.json', 'utf8'));

// Endpoint básico de "Hola Mundo"
app.get('/', (req, res) => {
    res.send('Hola Mundo');
});

// Endpoint para responder a preguntas
app.post('/ask-question', (req, res) => {
    const question = req.body.question;

    // Simplemente iterar y buscar similitud
    let bestMatch = null;
    let highestSimilarity = -1;

    for (const [pageNum, pageEmbeddings] of Object.entries(embeddings)) {
        pageEmbeddings.forEach(embedding => {
            const similarity = cosineSimilarity(question, embedding); // Comparar pregunta y embedding
            if (similarity > highestSimilarity) {
                highestSimilarity = similarity;
                bestMatch = { pageNum, embedding };
            }
        });
    }

    if (bestMatch) {
        res.json({ answer: `La respuesta más relevante se encuentra en la página ${bestMatch.pageNum}` });
    } else {
        res.json({ answer: "No se encontró una respuesta relevante." });
    }
});

// Función para calcular la similitud coseno
function cosineSimilarity(str1, str2) {
    const tokenizer = new natural.WordTokenizer();
    const tokens1 = tokenizer.tokenize(str1.toLowerCase());
    const tokens2 = tokenizer.tokenize(str2.toLowerCase());

    const vector1 = createVector(tokens1);
    const vector2 = createVector(tokens2);

    const dotProduct = vector1.reduce((sum, val, idx) => sum + val * vector2[idx], 0);
    const magnitude1 = Math.sqrt(vector1.reduce((sum, val) => sum + val * val, 0));
    const magnitude2 = Math.sqrt(vector2.reduce((sum, val) => sum + val * val, 0));

    return dotProduct / (magnitude1 * magnitude2);
}

// Crear un vector simple a partir de tokens
function createVector(tokens) {
    const vector = Array(10000).fill(0); // Ajusta el tamaño según lo necesites
    tokens.forEach(token => {
        const index = hashString(token) % 10000;
        vector[index] += 1;
    });
    return vector;
}

// Hash simple para asignar índices
function hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        hash = (hash << 5) - hash + str.charCodeAt(i);
        hash |= 0; // Convertir a 32-bit
    }
    return Math.abs(hash);
}

// Iniciar el servidor
app.listen(port, () => {
    console.log(`Servidor corriendo en http://localhost:${port}`);
});
