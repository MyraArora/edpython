import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import spacy

# Initialize Flask App
app = Flask(__name__)

# Load SpaCy model for NLP
nlp = spacy.load("en_core_web_sm")

# Load Dataset
data = pd.read_csv('questions_dataset.csv')  # Columns: 'question', 'category'

# Preprocessing Function
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Preprocess Questions
data['processed_question'] = data['question'].apply(preprocess_text)

# Encode Categories
label_encoder = LabelEncoder()
data['category_encoded'] = label_encoder.fit_transform(data['category'])

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    data['processed_question'], data['category_encoded'], test_size=0.2, random_state=42
)

# Build Model Pipeline
model = make_pipeline(TfidfVectorizer(), LinearSVC())
model.fit(X_train, y_train)

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get('question', '')
    if not user_input:
        return jsonify({'error': 'No question provided.'}), 400

    # Preprocess and Predict
    processed_input = preprocess_text(user_input)
    category_encoded = model.predict([processed_input])[0]
    category = label_encoder.inverse_transform([category_encoded])[0]

    # Predefined Responses
    responses = {
        "1": "Diabetes is a chronic medical condition that affects how your body processes glucose, the primary source of energy for cells. Glucose comes from the food you eat and is regulated in your bloodstream by a hormone called insulin, which is produced by the pancreas. In people with diabetes, this regulation is impaired due to either insufficient production of insulin, the body's inability to use insulin effectively, or a combination of both. This leads to elevated levels of glucose in the blood, a condition known as hyperglycemia. There are several types of diabetes. Type 1 diabetes is an autoimmune disorder where the immune system attacks and destroys the insulin-producing beta cells in the pancreas. This form of diabetes is typically diagnosed in childhood or early adulthood and requires daily insulin administration for survival. Type 2 diabetes, the most common form, occurs when the body becomes resistant to insulin or when the pancreas cannot produce enough insulin to maintain normal blood sugar levels. It is often associated with lifestyle factors such as poor diet, lack of exercise, obesity, and genetics. Gestational diabetes occurs during pregnancy when hormonal changes lead to insulin resistance, and while it usually resolves after childbirth, it increases the risk of developing Type 2 diabetes later in life. Common symptoms of diabetes include frequent urination, excessive thirst, extreme hunger, unexplained weight loss, fatigue, blurred vision, and slow healing of wounds. If left untreated, chronic high blood sugar levels can cause serious complications, including cardiovascular disease, kidney damage, nerve damage, and vision loss. Management of diabetes typically involves a combination of lifestyle changes, such as a healthy diet and regular exercise, along with medication or insulin therapy, depending on the type and severity of the condition. Regular monitoring of blood sugar levels is crucial to preventing complications and maintaining overall health.",
        "2": "Diabetes is a chronic medical condition characterized by high levels of sugar (glucose) in the blood due to issues with insulin production, insulin action, or both. The main types of diabetes are **Type 1 Diabetes**, **Type 2 Diabetes**, **Gestational Diabetes**, and less commonly, several other specific types. Type 1 Diabetes is an autoimmune condition where the immune system attacks insulin-producing beta cells in the pancreas, leading to little or no insulin production, and is typically diagnosed in childhood or adolescence, though it can occur at any age. Type 2 Diabetes is the most common form and occurs when the body becomes resistant to insulin or the pancreas cannot produce enough insulin; it is often associated with lifestyle factors like obesity, physical inactivity, and genetics. Gestational Diabetes develops during pregnancy and typically resolves after childbirth but increases the mother’s risk of developing Type 2 Diabetes later in life. Other types include **Monogenic Diabetes**, a rare genetic form of diabetes; **Cystic Fibrosis-Related Diabetes**, which occurs in people with cystic fibrosis due to damage to the pancreas; and **Secondary Diabetes**, caused by conditions such as pancreatitis or certain medications like steroids. Each type of diabetes requires specific management strategies to control blood sugar levels and prevent complications such as cardiovascular disease, nerve damage, kidney disease, and vision problems. Proper diagnosis is crucial for determining the appropriate treatment and lifestyle modifications.",
        "3": "Diabetes is caused by a combination of genetic, environmental, and lifestyle factors that disrupt the normal regulation of blood sugar levels in the body. In Type 2 Diabetes, the causes are multifactorial, including insulin resistance—where the body’s cells do not respond effectively to insulin—and a gradual decline in insulin production by the pancreas. Risk factors for Type 2 Diabetes include being overweight or obese, leading a sedentary lifestyle, a diet high in processed foods and sugar, genetic predisposition, and aging, although it can also develop in younger people. ",
        "4": "Diabetes is one of the most common chronic diseases globally, affecting hundreds of millions of people. According to the International Diabetes Federation (IDF), as of 2021, approximately **537 million adults aged 20-79** were living with diabetes, representing about 10.5% of the global population in this age group. This number is projected to rise to **643 million by 2030** and **783 million by 2045** if current trends continue. Type 2 Diabetes accounts for the vast majority (around 90-95%) of cases, while Type 1 Diabetes affects a smaller proportion of people, typically those diagnosed during childhood or adolescence. Diabetes is prevalent across all regions but is particularly common in low- and middle-income countries, where urbanization, sedentary lifestyles, and changes in diet have contributed to rising rates. Additionally, undiagnosed diabetes is a significant concern, with an estimated **240 million adults worldwide** unaware that they have the condition. Gestational diabetes affects up to 1 in 6 pregnancies globally, highlighting its impact on maternal and child health. The increasing prevalence of diabetes poses a major public health challenge, as it is a leading cause of complications such as heart disease, kidney failure, and amputations, making awareness, prevention, and effective management critical.",
        "5": "Type 1 and Type 2 diabetes are both chronic conditions that affect how the body regulates blood sugar, but they have several key differences in terms of their causes, development, and treatment. **Type 1 Diabetes** is an autoimmune condition where the immune system mistakenly attacks and destroys the insulin-producing beta cells in the pancreas, leading to little or no insulin production. It is most commonly diagnosed in childhood or adolescence, although it can occur at any age. People with Type 1 Diabetes must take insulin daily to regulate their blood sugar, as their bodies can no longer produce it naturally. The exact cause of this immune attack is not fully understood, but it is believed to involve genetic and environmental factors, such as viral infections. **Type 2 Diabetes**, on the other hand, is primarily a result of **insulin resistance**, where the body's cells do not respond properly to insulin. Over time, the pancreas may also produce less insulin. Type 2 Diabetes is much more common than Type 1 and is typically diagnosed in adulthood, though it is increasingly being seen in children due to rising obesity rates. It is strongly associated with lifestyle factors such as obesity, lack of physical activity, and poor diet, along with genetic predisposition. While insulin may be required in later stages of Type 2 Diabetes, many people can initially manage the condition with lifestyle changes, such as weight loss, a healthy diet, and exercise, as well as oral medications that help the body use insulin more effectively. Another key difference is that **Type 1 Diabetes** is usually not preventable, while **Type 2 Diabetes** is largely preventable through lifestyle changes, particularly in those at risk due to family history or obesity. Both types of diabetes, if not well-managed, can lead to serious complications, such as heart disease, nerve damage, kidney problems, and vision loss.",
        "6": "Gestational diabetes is a type of diabetes that develops during pregnancy, typically around the 24th to 28th week, and affects how the body processes glucose (sugar). During pregnancy, the placenta produces hormones that can make the body's cells less responsive to insulin, a condition known as **insulin resistance**. In some women, the pancreas cannot produce enough insulin to overcome this resistance, leading to high blood sugar levels. While gestational diabetes usually resolves after childbirth, it poses risks to both the mother and the baby during pregnancy. For the mother, it increases the risk of high blood pressure, preeclampsia, and developing Type 2 diabetes later in life. For the baby, gestational diabetes can lead to excessive birth weight, early delivery, respiratory issues, and an increased risk of developing obesity and Type 2 diabetes in childhood or adulthood. Factors that increase the risk of gestational diabetes include being overweight or obese, having a family history of diabetes, being older than 25, or having had gestational diabetes in a previous pregnancy. While it often has no obvious symptoms, gestational diabetes is typically diagnosed through routine blood tests during pregnancy. Treatment usually involves managing blood sugar levels through a healthy diet, regular physical activity, and sometimes insulin or oral medications if necessary. After delivery, women who have had gestational diabetes are monitored for Type 2 diabetes, as they are at a higher risk of developing it in the future.",
        "7": "Prediabetes is a condition where blood sugar levels are higher than normal but not high enough to be classified as Type 2 diabetes. It is a warning sign that a person is at increased risk of developing Type 2 diabetes, heart disease, and stroke. The condition often has no obvious symptoms, which is why many people with prediabetes may not realize they have it. The primary cause of prediabetes is **insulin resistance**, where the body's cells do not respond effectively to insulin, leading to higher blood sugar levels. Over time, the pancreas may struggle to produce enough insulin to keep blood sugar levels in check. Key risk factors for prediabetes include being overweight or obese, being physically inactive, having a family history of diabetes, having high blood pressure or abnormal cholesterol levels, and being over the age of 45. Additionally, women who had gestational diabetes are also at higher risk of developing prediabetes. Prediabetes can often be reversed through lifestyle changes such as losing weight, increasing physical activity, and adopting a healthier diet, focusing on whole foods, and reducing processed sugars. If left untreated, prediabetes can progress to Type 2 diabetes, so early detection and intervention are important for reducing the risk of developing full-blown diabetes. Regular blood tests, such as the fasting blood sugar test or an A1C test, can help detect prediabetes and provide an opportunity to take action before the condition advances.",
        "8": "Response for category 2.",
        "2": "Response for category 2.",
        "2": "Response for category 2.",
        "2": "Response for category 2.",
        "2": "Response for category 2.",
        "2": "Response for category 2.",
        "2": "Response for category 2.",
        "2": "Response for category 2.",
        "2": "Response for category 2.",
        "2": "Response for category 2.",    
        "2": "Response for category 2.",
        "2": "Response for category 2.",
        "2": "Response for category 2.",
        "2": "Response for category 2.",
        "2": "Response for category 2.",
        "2": "Response for category 2.",
        # ... Add all 100 responses
    }

    response = responses.get(category, "Sorry, I don't understand your question.")
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
