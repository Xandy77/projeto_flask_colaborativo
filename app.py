from flask import Flask, render_template, render_template_string, request
from jinja2 import TemplateNotFound
from service.github import obter_dados_github 
from dados import dados_personalizados
import pickle

app = Flask(__name__)
usuarios = ["lucianolpsf", "fernandallobao", "jesieldossantos", "Victorrezende19", "calebegomes740", "CaioHarrys", "aucelio0", "brunofluna", "Rafael-ai13", "Xandy77", "pauloalvezz" ] 


@app.route("/")
def home():
    membros = [obter_dados_github(usuario) for usuario in usuarios]
    return render_template("home.html", membros=membros)


@app.route("/<usuario>")
def rota_usuario(usuario):
    return render_template(f"{usuario.lower()}.html")


@app.route("/lucianolpsf/fruta", methods=['POST'])
def pred_fruta():
    peso = int(request.form['peso'])
    textura =int(request.form['textura'])

    with open('./analises/luciano/modelo_fruta.pkl', 'rb') as file:

        modelo = pickle.load(file)

    fruta =modelo.predict([[peso, textura]])
    return render_template_string(f'sua fruta é: {fruta[0]}')


@app.route("/CalebeGomes740/Predição-de-Diabetes-Calebe", methods=['POST'])
def pred_diabetCa():
    age = int(request.form['age'])
    gender = int(request.form['gender'])
    bmi = float(request.form['bmi']) 
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    smoking_history_str = request.form['smoking_history']
    smoking_history_map = {
        'never': 0,
        'no_info': 1,
        'current': 2,
        'ever': 3,
        'formerly': 4,
        'not_current': 5
    }
    smoking_history = smoking_history_map.get(smoking_history_str, -1)

    hba1c_level = float(request.form['hba1c_level'])
    blood_glucose_level = float(request.form['blood_glucose_level'])
    

    with open('./analises/calebe/predict_diabets.pkl', 'rb') as file:

        modelo = pickle.load(file)

    features = [[
            age, gender, bmi, hypertension, heart_disease,
            smoking_history, hba1c_level, blood_glucose_level
        ]]
    diabet_predic = modelo.predict(features)
    diabetes_status = diabet_predic[0] # Armazenamos o 0 ou 1 aqui

    # 3. Resposta Aprimorada (renderizando o template HTML)
    if diabetes_status == 0:
        message = "Não há indícios de diabetes."
    else:
        message = "Há indícios de diabetes. Por favor, consulte um médico."

    # Passamos a mensagem e o status para o template HTML
    return render_template('resul_pred_calebe.html', message=message, diabetes_status=diabetes_status)

@app.route('/xandy77/bmi_svm', methods=['POST'])
def predict():
    # Pega dados do formulário
    age_str = request.form.get('age')
    gender = request.form.get('gender')
    bmi_str = request.form.get('bmi')

    # Verifica se todos os dados foram fornecidos
    if not all([age_str, gender, bmi_str]):
        return render_template_string("Erro: Todos os campos (idade, gênero, BMI) são obrigatórios.")

    # Conversão de dados
    try:
        age = int(age_str)
        bmi = float(bmi_str)
    except (ValueError, TypeError):
        return render_template_string("Erro: Idade ou BMI inválidos. Por favor, insira valores numéricos.")

    # Conversão de gênero (exemplo: 'male' = 0, 'female' = 1)
    gender_map = {'male': 0, 'female': 1}
    gender = gender.lower()
    if gender not in gender_map:
        return render_template_string("Erro: Gênero inválido. Use 'male' ou 'female'.")
    
    gender_encoded = gender_map[gender]

    # Carrega o modelo treinado
    try:
        with open('./analises/xandy77/diabetes_predict_data.csv.pkl', 'rb') as file:
            modelo = pickle.load(file)
    except FileNotFoundError:
        return render_template_string("Erro: Arquivo do modelo não encontrado.")
    except Exception as e:
        return render_template_string(f"Erro ao carregar modelo: {str(e)}")

    # Faz a predição
    try:
        predict_diabete = modelo.predict([[age, gender_encoded, bmi]])
    except Exception as e:
        return render_template_string(f"Erro durante a predição: {str(e)}")

    # Retorna o resultado
    return render_template_string(
        f"Dados recebidos:<br>Idade = {age}<br>Gênero = {gender}<br>BMI = {bmi}<br><br><strong>Resultado da predição: {predict_diabete[0]}</strong>"
    )

if __name__ == '__main__':
    app.run(debug=True)