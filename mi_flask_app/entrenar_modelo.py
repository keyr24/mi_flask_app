from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

textos = ["me encanta este producto", "horrible, no me gustó", "excelente", "terrible experiencia"]
etiquetas = ["positivo", "negativo", "positivo", "negativo"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(textos)
modelo = MultinomialNB()
modelo.fit(X, etiquetas)

joblib.dump(modelo, 'modelo.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
