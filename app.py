from flask import Flask, render_template, request, jsonify
import json, random, string

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# ================= LOAD INTENTS =================
with open("intents.json", encoding="utf-8") as f:
    data = json.load(f)

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return stemmer.stem(text)

sentences, labels, responses = [], [], {}

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(clean_text(pattern))
        labels.append(intent["tag"])
    responses[intent["tag"]] = intent["responses"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)

model = MultinomialNB()
model.fit(X, labels)

# ================= DATA TOKO =================
menu_makanan = {
    "nasi goreng": 15000,
    "ayam geprek": 18000,
    "mie ayam": 12000,
    "bakso": 12000
}

menu_minuman = {
    "es teh": 5000,
    "es jeruk": 7000,
    "kopi": 6000
}

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json["message"].lower()

    # ===== LIST MENU MAKANAN =====
    if any(k in user_msg for k in ["menu makanan", "makanan apa", "makan apa"]):
        menu = "\n".join([f"- {m.title()}" for m in menu_makanan])
        return jsonify({"reply": f"Menu makanan kami:\n{menu}"})

    # ===== LIST MENU MINUMAN =====
    if any(k in user_msg for k in ["menu minuman", "minuman apa"]):
        menu = "\n".join([f"- {m.title()}" for m in menu_minuman])
        return jsonify({"reply": f"Menu minuman kami:\n{menu}"})

    # ===== HARGA PER ITEM =====
    for item, harga in {**menu_makanan, **menu_minuman}.items():
        if item in user_msg:
            return jsonify({"reply": f"{item.title()} harganya {harga//1000}k."})

    # ===== ML INTENT =====
    clean = clean_text(user_msg)
    vec = vectorizer.transform([clean])
    intent = model.predict(vec)[0]

    reply = random.choice(responses.get(intent, ["Maaf, saya belum paham."]))
    return jsonify({"reply": reply})

if __name__ == "__main__":
    print("APP.PY DIEKSEKUSI")
    app.run(debug=True)
