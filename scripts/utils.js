// Función para calcular la similitud coseno entre dos cadenas de texto
function cosineSimilarity(str1, str2) {
    const vector1 = stringToVector(str1);
    const vector2 = stringToVector(str2);
    
    const dotProduct = vector1.reduce((sum, val, idx) => sum + val * vector2[idx], 0);
    const magnitude1 = Math.sqrt(vector1.reduce((sum, val) => sum + val * val, 0));
    const magnitude2 = Math.sqrt(vector2.reduce((sum, val) => sum + val * val, 0));
    
    return dotProduct / (magnitude1 * magnitude2);
}

// Función para convertir una cadena de texto en un vector numérico (basado en el código de caracteres)
function stringToVector(str) {
    const vector = Array(256).fill(0); // Tamaño fijo de vector
    for (let i = 0; i < str.length; i++) {
        vector[str.charCodeAt(i) % 256] += 1; // Incrementar el índice correspondiente en el vector
    }
    return vector;
}

module.exports = {
    cosineSimilarity
};
