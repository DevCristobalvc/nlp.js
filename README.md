# Proyecto SWEBOK PDF Query

Este proyecto consiste en un sistema para extraer texto de un archivo PDF (SWEBOK v3) y generar embeddings basados en el contenido del texto. Luego, se utiliza NLP.js para responder preguntas relacionadas con el contenido de ese PDF.

## Estructura del Proyecto

```
/
├── data/                     # Contiene el archivo PDF y los resultados generados (JSON)
│   ├── swebok-v3.pdf         # El archivo PDF de entrada (SWEBOK v3)
│   ├── pdf_text.json         # Texto extraído del PDF en formato JSON
│   └── embeddings.json       # Embeddings generados para el texto extraído
├── scripts/                  # Scripts para extraer texto y generar embeddings
│   ├── extract_text.py       # Script para extraer texto del PDF
│   ├── generate_embeddings.py # Script para generar embeddings usando NLP.js
│   └── ask_question.py       # Script para realizar preguntas y obtener respuestas
├── server/                   # Servidor Express que maneja las solicitudes
│   └── app.js                # Archivo principal de servidor Express
├── .gitignore                # Archivos y carpetas ignorados por Git
└── README.md                 # Este archivo de documentación
```

## Instalación

### Requisitos

1. **Node.js**: Necesitarás Node.js para ejecutar el servidor Express y usar NLP.js.
2. **Python**: Python 3.x es necesario para ejecutar los scripts de extracción de texto.

### Pasos para instalar

1. Clona este repositorio:
   ```
   git clone https://github.com/tu-usuario/swebok-pdf-query.git
   ```

2. Instala las dependencias de Node.js:
   ```
   cd server
   npm install
   ```

3. Instala las dependencias de Python (si usas un entorno virtual):
   ```
   cd scripts
   pip install -r requirements.txt
   ```

4. Prepara el PDF (`swebok-v3.pdf`) y colócalo en la carpeta `data/`.

### Cómo ejecutar el proyecto

#### 1. Iniciar el servidor Express

En el directorio `server`, ejecuta el siguiente comando para iniciar el servidor:

```
node app.js
```

El servidor debería estar disponible en `http://localhost:3000`.

#### 2. Extraer texto del PDF

En el directorio `scripts`, ejecuta el siguiente comando para extraer el texto del PDF a un archivo JSON:

```
python extract_text.py
```

Esto generará un archivo `pdf_text.json` dentro de la carpeta `data/`.

#### 3. Generar embeddings

Usa el siguiente script para generar los embeddings basados en el texto extraído:

```
python generate_embeddings.py
```

Esto generará el archivo `embeddings.json` dentro de la carpeta `data/`.

#### 4. Realizar preguntas

Una vez que tengas los embeddings generados, puedes hacer preguntas enviando solicitudes POST a `http://localhost:3000/ask-question`. Asegúrate de enviar las preguntas y los embeddings en el cuerpo de la solicitud.

Ejemplo de cuerpo de solicitud:

```
{
  "question": "What is Software Engineering?",
  "embeddings": {
    "1": ["definition", "software engineering"]
  }
}
```

El servidor responderá con una respuesta basada en los embeddings generados.

## Contribución

Si deseas contribuir a este proyecto, por favor sigue estos pasos:

1. Haz un fork de este repositorio.
2. Crea una rama para tu cambio (`git checkout -b nueva-funcionalidad`).
3. Realiza tus cambios y haz commit (`git commit -am 'Agregada nueva funcionalidad'`).
4. Haz push a la rama (`git push origin nueva-funcionalidad`).
5. Abre un pull request.

## Licencia

Este proyecto está bajo la Licencia MIT - consulta el archivo [LICENSE](LICENSE) para más detalles.
