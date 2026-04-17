# Structured Outlines

Una biblioteca minimalista y sin dependencias externas para la **generación estructurada** con Modelos de Lenguaje Grandes (LLMs). 

Este proyecto está diseñado específicamente para entornos restrictivos (como Google Colab o servidores empresariales cerrados) donde no es posible instalar bibliotecas externas complejas, pero se necesita forzar a un LLM a responder con un formato JSON válido basado en un esquema de Pydantic.

## ✨ Características

* **Cero dependencias pesadas:** Utiliza únicamente la biblioteca estándar de Python y funciona de forma nativa con `transformers` de Hugging Face y `torch`.
* **Fácil de auditar:** Con apenas ~700 líneas de código, puedes leer, entender y auditar toda la lógica matemática y de procesamiento de texto en cuestión de minutos.
* **Generación garantizada:** Convierte esquemas JSON en expresiones regulares (Regex), luego las compila en Autómatas Finitos Deterministas (DFA) y finalmente filtra los tokens a nivel de logits (`LogitsProcessor`) para garantizar que la salida del modelo sea 100% válida.
* **Soporte de Pydantic:** Integración directa con los modelos de Pydantic (`BaseModel`) para definir la estructura deseada.

## ⚙️ ¿Cómo funciona?

El pipeline interno de la biblioteca sigue estos pasos de forma transparente:
1.  **`json_schema.py`**: Convierte el esquema JSON de tu modelo de Pydantic en una expresión regular compatible.
2.  **`regex_parser.py`**: Parsea la expresión regular generada y la transforma internamente (AST -> NFA -> DFA).
3.  **`guide.py`**: Crea una guía de transición de estados que mapea qué tokens del vocabulario del tokenizador son válidos en cada estado del DFA.
4.  **`generate.py`**: Implementa el `StructuredLogitsProcessor` que se inyecta en la función `.generate()` de Hugging Face, enmascarando en cada paso los tokens que romperían la estructura del JSON.

## 🚀 Uso Rápido
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
from structured_outlines import generate

# 1. Define tu estructura de salida con Pydantic
class Respuesta(BaseModel):
    nombre: str
    edad: int
    es_estudiante: bool

# 2. Carga tu modelo y tokenizador de Hugging Face
model_id = "tu-modelo-favorito" # ej. "Qwen/Qwen1.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# 3. Genera la respuesta estructurada
prompt = "Extrae la información: Juan tiene 25 años y estudia en la universidad.\n"

resultado_json = generate(
    model=model,
    tokenizer=tokenizer,
    schema=Respuesta,
    prompt=prompt,
    max_new_tokens=100
)

print(resultado_json)
# Salida esperada: {"nombre": "Juan", "edad": 25, "es_estudiante": true}
```
