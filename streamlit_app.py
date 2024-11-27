import streamlit as st
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Cargar el modelo y el tokenizador de GPT-2
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Asegurarse de que el modelo está en modo de evaluación
model.eval()

# Base de datos de FAQ con variaciones
faq_variations = {
    "título profesional": [
        "título profesional", 
        "solicitar título", 
        "cómo obtener el título", 
        "tramitar título profesional", 
        "dame información sobre mi título", 
        "requisitos para título"
    ],
    "certificado de egresado": [
        "certificado de egresado", 
        "cómo obtener certificado de egresado", 
        "solicitar certificado", 
        "requisitos para certificado"
    ],
    "constancia de estudios": [
        "constancia de estudios", 
        "cómo obtener constancia de estudios", 
        "solicitar constancia", 
        "requisitos para constancia"
    ],
    "requisitos para graduarse": [
        "requisitos para graduarse", 
        "cómo graduarse", 
        "cuáles son los requisitos para graduarme", 
        "qué necesito para graduarme"
    ],
    "certificado de notas": [
        "certificado de notas", 
        "cómo obtener certificado de notas", 
        "solicitar certificado de notas", 
        "requisitos para certificado de notas"
    ],
    "documentos para graduación": [
        "documentos para graduación", 
        "qué documentos necesito para graduarme", 
        "requisitos documentales para graduarse"
    ]
}

# Base de datos de respuestas
faq_db = {
    "título profesional": "El trámite para obtener tu título profesional se inicia en la Oficina de Egresados. Debes haber aprobado todos los cursos y presentar tu solicitud a través del sistema de la universidad.",
    "certificado de egresado": "Para solicitar tu certificado de egresado, debes llenar el formulario en la página web de la Oficina de Egresados y esperar 3 días hábiles para su entrega.",
    "constancia de estudios": "Puedes obtener una constancia de estudios solicitándola a través de tu cuenta en el portal de la universidad o en la Oficina de Egresados.",
    "requisitos para graduarse": "Para graduarte, debes haber cumplido con todos los requisitos académicos establecidos en el reglamento de la universidad. Consulta el portal de graduados para más detalles.",
    "certificado de notas": "Para obtener un certificado de notas, debes presentar una solicitud en línea a través del portal de egresados de la UNFV.",
    "documentos para graduación": "Los documentos necesarios para graduarte incluyen tu expediente académico, tu constancia de egresado, y tu solicitud de título, los cuales debes entregar en la Oficina de Egresados."
}

# Función de preprocesamiento para normalizar la entrada del usuario
def preprocess_input(user_input):
    user_input = user_input.lower()  # Convertir todo a minúsculas
    user_input = re.sub(r'\bdame\b|\bcomo\b|\bpara\b|\bsolicitar\b|\bquiero\b|\binformacion\b', '', user_input)  # Eliminar palabras innecesarias
    return user_input.strip()

# Función para obtener la respuesta del chatbot
def get_chatbot_response(user_input):
    user_input = preprocess_input(user_input)
    
    # Verificar si alguna de las variaciones de las claves del FAQ está en la entrada del usuario
    for key, variations in faq_variations.items():
        for variation in variations:
            if variation in user_input:  # Compara la variación con la entrada del usuario
                return faq_db[key]
    
    # Si no se encuentra ninguna coincidencia, generar una respuesta predeterminada
    return "Lo siento, no pude encontrar una respuesta exacta a tu pregunta. ¿Podrías reformularla?"

# Configuración de la página
st.set_page_config(page_title="Chatbot Administrativo UNFV", page_icon=":guardsman:", layout="wide")

# Encabezado principal
st.title("Chatbot Administrativo - Oficina de Egresados UNFV")
st.write("¡Hola! ¿En qué puedo ayudarte hoy con los trámites administrativos de la Universidad Nacional Federico Villarreal (UNFV)?")

# Dividir la pantalla en dos secciones: izquierda (chat) y derecha (entrada de texto y botones)
col1, col2 = st.columns([3, 1])

# Mantener historial de la conversación
if "history" not in st.session_state:
    st.session_state.history = []

# Recibir entrada del usuario
with col2:
    user_input = st.text_input("Escribe tu pregunta:", placeholder="Escribe aquí tu consulta sobre egresados...", key="input", max_chars=200)
    send_button = st.button("Enviar", key="send")

# Estilo visual para los mensajes
def style_message(message, sender):
    if sender == 'usuario':
        return f'<div style="background-color:#e3f2fd; padding: 10px; border-radius: 10px; margin: 5px; font-size: 16px; max-width: 70%; float: left;"> <strong>Tú:</strong> {message}</div>'
    else:
        return f'<div style="background-color:#fff9c4; padding: 10px; border-radius: 10px; margin: 5px; font-size: 16px; max-width: 70%; float: right;"> <strong>Chatbot:</strong> {message}</div>'

# Mostrar el chat en la columna izquierda
with col1:
    st.markdown("<hr>", unsafe_allow_html=True)  # Línea divisoria entre la cabecera y el chat

    # Mostrar el historial de mensajes
    for i, message in enumerate(st.session_state.history):
        decoded_message = tokenizer.decode(message[0], skip_special_tokens=True)
        if i % 2 == 0:  # Mensajes del usuario
            st.markdown(style_message(decoded_message, 'usuario'), unsafe_allow_html=True)
        else:  # Respuestas del modelo
            st.markdown(style_message(decoded_message, 'chatbot'), unsafe_allow_html=True)

# Enviar mensaje y generar respuesta
if send_button and user_input:
    response = get_chatbot_response(user_input)  # Obtiene la respuesta basada en las variaciones
    st.session_state.history.append([user_input])  # Guardar la pregunta del usuario en el historial
    st.session_state.history.append([response])  # Guardar la respuesta del chatbot en el historial

    # Mostrar la respuesta generada
    st.markdown(style_message(response, 'chatbot'), unsafe_allow_html=True)

    # Botón de retroalimentación
    feedback = st.radio("¿La respuesta fue útil?", ("Sí", "No"), key="feedback")
    if feedback == "No":
        st.write("Gracias por tu retroalimentación. Intentaremos mejorar nuestras respuestas.")

# Barra lateral con funciones adicionales
st.sidebar.title("Opciones adicionales")
st.sidebar.write("### ¿Necesitas ayuda más detallada?")
st.sidebar.write("Este chatbot está diseñado para brindarte información sobre los trámites administrativos de la UNFV. Si no encuentras lo que buscas, por favor contáctanos directamente.")
st.sidebar.write("Puedes intentar con algunas de estas preguntas frecuentes:")
st.sidebar.write("- ¿Cómo solicitar mi certificado de egresado?")
st.sidebar.write("- ¿Cuáles son los requisitos para graduarme?")
st.sidebar.button("Recargar la conversación", on_click=lambda: st.session_state.history.clear())
