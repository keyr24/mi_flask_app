from flask import Flask, render_template, request
import joblib

# Cargar modelo y vectorizador
modelo = joblib.load('modelo.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def create_app():
    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def index():
        resultado = ""
        if request.method == "POST":
            texto = request.form["texto"]
            vect_texto = vectorizer.transform([texto])
            prediccion = modelo.predict(vect_texto)
            resultado = f"Sentimiento: {prediccion[0]}"
        return render_template("index.html", resultado=resultado)

    return app
