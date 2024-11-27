import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Cargar el modelo y el tokenizador
model_name = "gpt2"  # Puedes cambiar a otro modelo como "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Asegurarse de que el modelo está en modo de evaluación
model.eval()

# Base de datos simple de preguntas frecuentes (FAQ)
faq_db = {
    "certificado de egresado": "Para solicitar tu certificado de egresado, debes llenar el formulario en la página web de la Oficina de Egresados y esperar 3 días hábiles para su entrega.",
    "título profesional": "El trámite para obtener tu título profesional se inicia en la Oficina de Egresados. Debes haber aprobado todos los cursos y presentar tu solicitud a través del sistema de la universidad.",
    "constancia de estudios": "Puedes obtener una constancia de estudios solicitándola a través de tu cuenta en el portal de la universidad o en la Oficina de Egresados.",
    "requisitos para graduarse": "Para graduarte, debes haber cumplido con todos los requisitos académicos establecidos en el reglamento de la universidad. Consulta el portal de graduados para más detalles.",
    "certificado de notas": "Para obtener un certificado de notas, debes presentar una solicitud en línea a través del portal de egresados de la UNFV.",
    "documentos para graduación": "Los documentos necesarios para graduarte incluyen tu expediente académico, tu constancia de egresado, y tu solicitud de título, los cuales debes entregar en la Oficina de Egresados."
}

# Función para generar respuesta
def get_chatbot_response(user_input, history=None):
    if history is None:
        history = []

    # Convertir la entrada del usuario a minúsculas para comparación
    user_input = user_input.lower()

    # Verificar si la entrada coincide con alguna de las preguntas frecuentes
    response = faq_db.get(user_input, None)

    if response is None:
        # Si no se encuentra en las preguntas frecuentes, usar el modelo general
        inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        history.append(inputs)

        with torch.no_grad():
            output = model.generate(torch.cat(history, dim=-1), max_length=300, pad_token_id=tokenizer.eos_token_id, num_return_sequences=1)

        response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Agregar un toque educativo y estructurado a la respuesta
    response = f"**¡Aquí tienes la información que necesitas!**\n\n{response}\n\nSi tienes más dudas, ¡no dudes en preguntar!"

    return response, history

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
    response, history = get_chatbot_response(user_input, st.session_state.history)
    st.session_state.history = history  # Guardar el historial en el estado de la sesión

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

