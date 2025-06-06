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


if __name__ == '__main__':
    app.run(debug=True)