# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn import svm
import json

app = Flask(__name__)

# Dados iniciais para treinamento - famílias Capuleto (0) e Montéquio (1)
capuletos = np.array([[1.2, 0.4], [1.5, 0.6], [1.8, 0.2], [1.6, 0.3], [1.4, 0.1], 
                      [1.3, 0.5], [1.7, 0.4], [1.6, 0.2], [1.3, 0.3], [1.5, 0.1]])

montequios = np.array([[4.5, 1.4], [4.2, 1.2], [4.0, 1.0], [4.3, 1.3], [4.1, 1.5],
                       [4.6, 1.2], [4.4, 1.4], [4.2, 1.1], [4.5, 1.0], [4.7, 1.3]])

# Configuração inicial do modelo SVM
# Criando conjunto de treinamento
X = np.vstack((capuletos, montequios))
y = np.array([0] * len(capuletos) + [1] * len(montequios))

# Treinando o modelo
modelo = svm.SVC(kernel='linear', C=0.1)
modelo.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dados')
def get_dados():
    # Obter coeficientes da fronteira de decisão
    w = modelo.coef_[0]
    b = modelo.intercept_[0]
    
    # Calcular pontos para desenhar a linha de decisão no gráfico
    # y = -(w[0]/w[1])*x - b/w[1]
    x_min, x_max = 0, 6
    y_min = -(w[0]/w[1])*x_min - b/w[1]
    y_max = -(w[0]/w[1])*x_max - b/w[1]

    # Calcular também as linhas de margem
    # Pegar alguns vetores de suporte
    support_vectors = modelo.support_vectors_
    
    # Prepare os dados para retornar ao frontend
    dados = {
        'capuletos': capuletos.tolist(),
        'montequios': montequios.tolist(),
        'linha_decisao': {
            'x': [float(x_min), float(x_max)],
            'y': [float(y_min), float(y_max)]
        },
        'vetores_suporte': support_vectors.tolist()
    }
    
    return jsonify(dados)

@app.route('/classificar', methods=['POST'])
def classificar():
    dados = request.get_json()
    novo_ponto = [dados['x'], dados['y']]
    
    # Classificar o ponto
    resultado = modelo.predict([novo_ponto])[0]
    familia = "Montéquio" if resultado == 1 else "Capuleto"
    
    # Calcular distância até a fronteira de decisão
    distancia = modelo.decision_function([novo_ponto])[0]
    
    return jsonify({
        'classe': int(resultado),
        'familia': familia,
        'distancia': float(distancia),
        'ponto': novo_ponto
    })

@app.route('/adicionar', methods=['POST'])
def adicionar():
    global capuletos, montequios, X, y, modelo
    
    dados = request.get_json()
    novo_ponto = [dados['x'], dados['y']]
    classe = dados['classe']
    
    # Adicionar ponto aos dados correspondentes
    if classe == 0:  # Capuleto
        capuletos = np.vstack((capuletos, novo_ponto))
    else:  # Montéquio
        montequios = np.vstack((montequios, novo_ponto))
    
    # Reconstruir conjunto de dados
    X = np.vstack((capuletos, montequios))
    y = np.array([0] * len(capuletos) + [1] * len(montequios))
    
    # Retreinar o modelo
    modelo = svm.SVC(kernel='linear', C=0.1)
    modelo.fit(X, y)
    
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)