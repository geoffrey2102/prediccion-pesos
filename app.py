from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import pandas as pd
import pickle
import numpy as np
import os

# Suprimir advertencias de TensorFlow y deshabilitar GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suprime advertencias (0 = todas, 3 = ninguna)
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Fuerza uso de CPU

app = Flask(__name__)

# Habilitar CORS para permitir solicitudes desde otros dominios
CORS(app)

# Configuración para archivos estáticos
app.static_folder = 'static'

# Crear carpeta static si no existe
os.makedirs(app.static_folder, exist_ok=True)

# Asegurarse de que index.html esté en la carpeta static
html_content = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Calcular Precio de Envío</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 600px;
            margin: 20px auto;
            padding: 20px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
        }
        input, select {
            width: 100%;
            padding: 8px;
            margin-bottom: 10px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        #resultado {
            margin-top: 20px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>Calcular Precio de Envío</h1>
    <div class="form-group">
        <label for="peso">Peso (kg):</label>
        <input type="number" id="peso" step="0.1" min="0" required>
    </div>
    <div class="form-group">
        <label for="inicio">Ciudad de Inicio:</label>
        <select id="inicio" required>
            <option value="">Seleccione una ciudad</option>
            <option value="Lima">Lima</option>
            <option value="Arequipa">Arequipa</option>
            <option value="Trujillo">Trujillo</option>
            <option value="Chiclayo">Chiclayo</option>
            <option value="Piura">Piura</option>
            <option value="Cusco">Cusco</option>
            <option value="Iquitos">Iquitos</option>
            <option value="Huancayo">Huancayo</option>
            <option value="Pucallpa">Pucallpa</option>
            <option value="Tacna">Tacna</option>
            <option value="Ayacucho">Ayacucho</option>
            <option value="Chimbote">Chimbote</option>
            <option value="Ica">Ica</option>
            <option value="Juliaca">Juliaca</option>
            <option value="Tarapoto">Tarapoto</option>
        </select>
    </div>
    <div class="form-group">
        <label for="llegada">Ciudad de Llegada:</label>
        <select id="llegada" required>
            <option value="">Seleccione una ciudad</option>
            <option value="Lima">Lima</option>
            <option value="Arequipa">Arequipa</option>
            <option value="Trujillo">Trujillo</option>
            <option value="Chiclayo">Chiclayo</option>
            <option value="Piura">Piura</option>
            <option value="Cusco">Cusco</option>
            <option value="Iquitos">Iquitos</option>
            <option value="Huancayo">Huancayo</option>
            <option value="Pucallpa">Pucallpa</option>
            <option value="Tacna">Tacna</option>
            <option value="Ayacucho">Ayacucho</option>
            <option value="Chimbote">Chimbote</option>
            <option value="Ica">Ica</option>
            <option value="Juliaca">Juliaca</option>
            <option value="Tarapoto">Tarapoto</option>
        </select>
    </div>
    <button onclick="calcularPrecio()">Calcular Precio</button>
    <div id="resultado"></div>

    <script>
        async function calcularPrecio() {
            const peso = document.getElementById('peso').value;
            const inicio = document.getElementById('inicio').value;
            const llegada = document.getElementById('llegada').value;
            const resultadoDiv = document.getElementById('resultado');

            if (!peso || !inicio || !llegada) {
                resultadoDiv.innerHTML = 'Por personally, complete todos los campos';
                return;
            }

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ peso, inicio, llegada })
                });

                console.log('Status:', response.status);
                console.log('Content-Type:', response.headers.get('Content-Type'));
                const text = await response.text();
                console.log('Response body:', text);

                try {
                    const data = JSON.parse(text);
                    if (response.ok) {
                        resultadoDiv.innerHTML = `Precio estimado: ${data.precio_predicho}`;
                    } else {
                        resultadoDiv.innerHTML = `Error: ${data.error || 'Error desconocido'}`;
                    }
                } catch (e) {
                    resultadoDiv.innerHTML = `Error: Respuesta no es JSON válido: ${text}`;
                }
            } catch (error) {
                resultadoDiv.innerHTML = `Error: ${error.message}`;
            }
        }
    </script>
</body>
</html>"""

# Guardar el HTML en el archivo estático
with open(os.path.join(app.static_folder, 'index.html'), 'w', encoding='utf-8') as f:
    f.write(html_content)

# Load model and encoders
try:
    # Manejar cambios de versiones en TensorFlow
    try:
        # Para TensorFlow 2.16+
        model = tf.keras.models.load_model('modelo_envios.h5', compile=False)
    except:
        # Para versiones anteriores
        model = tf.keras.models.load_model('modelo_envios.h5')
        
    with open('le_inicio.pkl', 'rb') as f:
        le_inicio = pickle.load(f)
    with open('le_llegada.pkl', 'rb') as f:
        le_llegada = pickle.load(f)
    with open('X_train_columns.pkl', 'rb') as f:
        X_train_columns = pickle.load(f)
    print("Modelo y codificadores cargados correctamente")
except Exception as e:
    print(f"Error al cargar modelo o codificadores: {str(e)}")
    # Verificar si los archivos existen
    print(f"Archivo modelo_envios.h5 existe: {os.path.exists('modelo_envios.h5')}")
    print(f"Archivo le_inicio.pkl existe: {os.path.exists('le_inicio.pkl')}")
    print(f"Archivo le_llegada.pkl existe: {os.path.exists('le_llegada.pkl')}")
    print(f"Archivo X_train_columns.pkl existe: {os.path.exists('X_train_columns.pkl')}")

# List of cities (matching training data)
ciudades = ['Lima', 'Arequipa', 'Trujillo', 'Chiclayo', 'Piura',
           'Cusco', 'Iquitos', 'Huancayo', 'Pucallpa', 'Tacna',
           'Ayacucho', 'Chimbote', 'Ica', 'Juliaca', 'Tarapoto']

@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(f"Datos recibidos: {data}")
        if not data:
            print("Error: No se recibieron datos JSON")
            return jsonify({'error': 'No se recibieron datos JSON'}), 400
            
        peso = float(data.get('peso', 0))
        inicio = data.get('inicio', '')
        llegada = data.get('llegada', '')

        print(f"Peso: {peso}, Inicio: {inicio}, Llegada: {llegada}")

        if not all([peso, inicio, llegada]):
            print("Error: Faltan datos requeridos")
            return jsonify({'error': 'Faltan datos requeridos'}), 400
        if inicio not in ciudades or llegada not in ciudades:
            print("Error: Ciudad no válida")
            return jsonify({'error': 'Ciudad no válida'}), 400
        if peso <= 0:
            print("Error: El peso debe ser mayor que 0")
            return jsonify({'error': 'El peso debe ser mayor que 0'}), 400

        nuevo_envio = pd.DataFrame({
            'Peso': [peso],
            'Inicio': [inicio],
            'Llegada': [llegada]
        })

        print("Transformando ciudades...")
        nuevo_envio['Inicio_encoded'] = le_inicio.transform(nuevo_envio['Inicio'])
        nuevo_envio['Llegada_encoded'] = le_llegada.transform(nuevo_envio['Llegada'])

        nuevo_inicio_onehot = pd.get_dummies(nuevo_envio['Inicio'], prefix='Inicio')
        nuevo_llegada_onehot = pd.get_dummies(nuevo_envio['Llegada'], prefix='Llegada')

        for col in X_train_columns:
            if col not in nuevo_inicio_onehot.columns and 'Inicio' in col:
                nuevo_inicio_onehot[col] = 0
            if col not in nuevo_llegada_onehot.columns and 'Llegada' in col:
                nuevo_llegada_onehot[col] = 0

        nuevo_X = pd.concat([nuevo_envio[['Peso']], nuevo_inicio_onehot, nuevo_llegada_onehot], axis=1)
        print(f"Columnas de nuevo_X: {nuevo_X.columns.tolist()}")

        nuevo_X = nuevo_X[X_train_columns]
        print("Realizando predicción...")
        prediccion = model.predict(nuevo_X, verbose=0)
        precio_predicho = float(prediccion[0][0])
        print(f"Predicción: {precio_predicho}")

        return jsonify({'precio_predicho': f'{precio_predicho:.2f} soles'})

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error en la predicción: {str(e)}\n{error_trace}")
        return jsonify({'error': f'Error en la predicción: {str(e)}'}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))  # Usa el puerto de Render o 5000 por defecto
    app.run(debug=False, host='0.0.0.0', port=port)