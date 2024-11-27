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
        "8": "Currently, there is no cure for diabetes, but the condition can be effectively managed to prevent complications and improve quality of life. The approach to managing diabetes depends on the type. **Type 1 Diabetes**, which is an autoimmune condition where the body cannot produce insulin, requires lifelong insulin therapy to regulate blood sugar levels. While Type 1 Diabetes cannot be cured, people with the condition can live healthy lives by carefully managing their blood sugar through insulin injections, diet, exercise, and continuous blood sugar monitoring. **Type 2 Diabetes**, which is primarily caused by insulin resistance and a decline in insulin production, may be more manageable and, in some cases, reversible through lifestyle changes. People with Type 2 Diabetes can often control their blood sugar levels by making changes such as losing weight, increasing physical activity, and adopting a healthy diet. In some instances, if Type 2 Diabetes is detected early and lifestyle changes are made, blood sugar levels can return to normal, and medication may no longer be necessary, but this is not considered a cure as the risk of developing diabetes again remains. Medical treatments, including oral medications and insulin, may still be required for many people with Type 2 Diabetes, particularly as the condition progresses. **Gestational Diabetes** usually resolves after childbirth, but women who experience it are at a higher risk of developing Type 2 Diabetes later in life. There are also ongoing efforts in medical research to find potential cures, such as pancreatic transplants, stem cell therapy, and immunotherapy for Type 1 Diabetes, but these are still in experimental stages. Managing diabetes through regular monitoring, medication, and lifestyle changes is crucial to prevent complications like heart disease, kidney failure, and vision loss.",
        "9": "Diabetes can have a significant impact on life expectancy, primarily due to the complications that arise from poorly managed blood sugar levels over time. While many people with diabetes live long, healthy lives if their condition is well-controlled, uncontrolled or poorly managed diabetes can lead to a range of serious health problems that may shorten life expectancy. **Type 1 Diabetes**, which requires lifelong insulin therapy, can increase the risk of complications such as cardiovascular disease, kidney failure, nerve damage, and vision loss. Advances in diabetes care have significantly improved outcomes for those with Type 1 Diabetes, but they still face a higher risk of these complications compared to the general population, which can reduce life expectancy by around 10 to 15 years if not carefully managed. **Type 2 Diabetes**, which is often linked to lifestyle factors such as obesity, poor diet, and lack of exercise, is also associated with a higher risk of heart disease, stroke, kidney disease, and other complications. If Type 2 Diabetes is poorly controlled, it can shorten life expectancy by about 5 to 10 years, though this can vary depending on the individual’s ability to manage their blood sugar, blood pressure, and cholesterol levels, as well as whether they develop other health conditions like high blood pressure or heart disease. **Gestational Diabetes** does not directly affect life expectancy, as it usually resolves after childbirth, but women who experience gestational diabetes are at an increased risk of developing Type 2 Diabetes later in life, which could impact their long-term health and life expectancy. Additionally, factors such as early detection, lifestyle modifications (such as maintaining a healthy diet, regular exercise, and weight management), and adherence to medication regimens can help reduce the risks and complications associated with diabetes, potentially leading to a life expectancy similar to that of people without diabetes. Preventing or managing complications through regular monitoring and proactive care is essential for improving life expectancy for people with diabetes.",
        "10": "The global prevalence of diabetes has been steadily increasing over the past few decades, and this trend is expected to continue. According to the International Diabetes Federation (IDF), by 2021, approximately 537 million adults worldwide (aged 20-79) were living with diabetes, representing about 10.5% of the global population in this age group. This number is projected to rise to 643 million by 2030 and to 783 million by 2045 if current trends persist. Type 2 Diabetes accounts for the majority of cases, driven largely by lifestyle factors such as poor diet, lack of physical activity, and obesity, which are becoming more prevalent globally, particularly in low- and middle-income countries. The rise in diabetes cases is also closely linked to urbanization, as more people adopt sedentary lifestyles and diets high in processed foods, which contribute to insulin resistance and the development of Type 2 Diabetes. There is a notable shift in diabetes demographics, with the disease previously being more common in high-income countries but now spreading rapidly in low- and middle-income countries, where healthcare systems may not be as equipped to address the rising burden of the disease. Additionally, the prevalence of **prediabetes**—a condition where blood sugar levels are higher than normal but not yet in the diabetic range—is also increasing, further contributing to the rise in diabetes rates. **Gestational diabetes** is also on the rise, with approximately 1 in 6 pregnancies globally affected, which highlights the growing concern about maternal health and the long-term risks for both mothers and children. One of the significant global challenges is the increasing number of undiagnosed diabetes cases. It is estimated that nearly half of people with diabetes worldwide do not know they have the condition. This underscores the need for better awareness, early detection, and intervention. In response to the growing prevalence of diabetes, there is a heightened focus on prevention strategies, including promoting healthy lifestyles, increasing access to medical care, and improving public health initiatives.",
        "11": "The typical signs of diabetes can vary depending on the type and how advanced the condition is, but some common symptoms include frequent urination, which occurs as high blood sugar levels cause the kidneys to work harder to filter and absorb excess glucose, leading to more urination, especially at night. Increased thirst is another common sign as frequent urination causes the body to lose more water, leading to dehydration and causing intense thirst. People with diabetes may also experience extreme hunger, as their bodies cannot properly use the glucose from food for energy. Fatigue is another common symptom due to the inability to efficiently use glucose for energy. Blurred vision can happen when high blood sugar pulls fluid from tissues, including the eyes, affecting the ability to focus. Slow-healing sores or infections can occur because diabetes can impair the body’s ability to heal. Unexplained weight loss, especially in Type 1 diabetes, can happen when the body starts breaking down muscle and fat for energy. Tingling or numbness in the hands or feet, known as neuropathy, can be a result of nerve damage from high blood sugar levels. Dark patches of skin in areas like the armpits, neck, or groin, known as acanthosis nigricans, can occur in people with insulin resistance, which is common in Type 2 diabetes. Frequent infections can also be a sign, as elevated blood sugar weakens the immune system. If any of these symptoms are noticed, it’s important to consult a healthcare provider for a proper diagnosis and blood tests, as early detection and management can help prevent complications.",
        "12": "Diabetes is diagnosed through several tests that measure blood sugar levels. The most common tests used are the **fasting blood sugar test**, the **oral glucose tolerance test (OGTT)**, and the **A1C test**. The **fasting blood sugar test** involves measuring blood sugar after an overnight fast, and a reading of 126 mg/dL or higher on two separate occasions indicates diabetes. If the blood sugar level is between 100 and 125 mg/dL, it is considered prediabetes. The **oral glucose tolerance test (OGTT)** measures blood sugar levels after fasting and then drinking a sugary solution. A blood sugar level of 200 mg/dL or higher two hours after drinking the solution indicates diabetes, while a level between 140 and 199 mg/dL suggests prediabetes. The **A1C test** measures the average blood sugar level over the past two to three months, with a result of 6.5% or higher confirming a diagnosis of diabetes. An A1C between 5.7% and 6.4% indicates prediabetes. In addition to these tests, doctors may also perform **random blood sugar tests** to diagnose diabetes if symptoms are present. A blood sugar level of 200 mg/dL or higher, regardless of when the person last ate, is diagnostic of diabetes. Diagnosis may involve a combination of these tests to confirm the presence of diabetes or prediabetes, and follow-up tests are often done to monitor blood sugar levels over time.",
        "13": "The HbA1c test, also known as the A1C test, measures the average blood sugar (glucose) level over the past two to three months. It reflects how well your blood sugar has been controlled during that time by measuring the percentage of hemoglobin— a protein in red blood cells— coated with sugar. Since red blood cells live for about 120 days, the A1C test provides a long-term view of blood sugar levels. A result below 5.7% is considered normal, while an A1C between 5.7% and 6.4% indicates prediabetes. An A1C of 6.5% or higher on two separate tests confirms a diagnosis of diabetes. The A1C test is used both for diagnosing diabetes and for monitoring blood sugar control in people who already have the condition.",
        "14": "Normal blood sugar levels vary depending on when the test is taken. For a **fasting blood sugar test**, which measures blood sugar after not eating for at least 8 hours, normal levels are between **70 and 99 mg/dL**. For a **random blood sugar test**, which can be done at any time of day, a normal level is typically **less than 140 mg/dL**. The **oral glucose tolerance test (OGTT)** is used to check how the body processes sugar. For this test, a normal level is **less than 140 mg/dL** two hours after drinking a sugary solution. An **A1C test**, which shows the average blood sugar over the past two to three months, should be less than **5.7%** for normal levels. If blood sugar levels are higher than these ranges, it could indicate prediabetes or diabetes.",
        "15": "Symptoms of low blood sugar, or hypoglycemia, can vary from mild to severe and may include shakiness, sweating, rapid heartbeat, dizziness, confusion, irritability, hunger, weakness, and difficulty concentrating. As blood sugar continues to drop, more severe symptoms can develop, such as blurred vision, headaches, numbness or tingling in the lips or tongue, and in extreme cases, loss of consciousness or seizures. If someone experiences symptoms of hypoglycemia, it's important to quickly consume a fast-acting source of sugar, such as glucose tablets, juice, or candy, to raise blood sugar levels and prevent further complications.",
        "16": "Symptoms of high blood sugar, or hyperglycemia, can include frequent urination, excessive thirst, dry mouth, fatigue, blurred vision, and headaches. People may also experience unexplained weight loss, increased hunger, and slow-healing wounds or infections. In severe cases, hyperglycemia can lead to more serious complications such as diabetic ketoacidosis (DKA) or hyperosmolar hyperglycemic state (HHS), which can cause confusion, nausea, vomiting, abdominal pain, and in extreme cases, loss of consciousness. If blood sugar remains high over time, it can increase the risk of complications such as heart disease, nerve damage, and kidney problems. It is important to monitor blood sugar levels and manage them effectively to prevent long-term damage.",
        "17": "Yes, diabetes can be asymptomatic, especially in the early stages. Many people with Type 2 diabetes, in particular, may not experience noticeable symptoms for years, even though their blood sugar levels are elevated. This is why diabetes is often undiagnosed until it causes complications or is detected during routine screening. Some individuals may have only mild symptoms, such as fatigue or increased thirst, which they may not immediately associate with diabetes. It is important to have regular check-ups and screenings, especially if you have risk factors for diabetes, to catch the condition early and manage it effectively before complications arise.",    
        "18": "Gestational diabetes is typically detected through an **oral glucose tolerance test (OGTT)**, usually between the 24th and 28th week of pregnancy. The process involves the following steps: first, a blood sample is taken after the woman has fasted for at least 8 hours. Then, she drinks a sugary solution containing a specific amount of glucose, and blood samples are taken at intervals, usually one hour, two hours, and sometimes three hours after consuming the drink. If the blood sugar levels are higher than normal at any of these points, it suggests gestational diabetes. The specific threshold values may vary slightly depending on the guidelines used, but if two or more of the blood sugar readings exceed the designated limits, a diagnosis of gestational diabetes is made. This test helps doctors identify women who may be at risk of gestational diabetes so that they can be monitored and treated accordingly to prevent complications for both the mother and baby.",
        "19": "Family history plays a significant role in diagnosing diabetes, as having a close relative with diabetes increases the risk of developing the condition. In particular, a family history of Type 1 or Type 2 diabetes can make someone more likely to develop diabetes themselves. For **Type 2 diabetes**, the risk is especially high if parents or siblings have the condition, as genetic factors contribute to insulin resistance and impaired insulin production. In **Type 1 diabetes**, while the role of genetics is less understood, having a first-degree relative with the condition can still increase the risk, though environmental factors also play a part. A family history of **gestational diabetes** can also increase the risk for women of developing diabetes later in life. Because of the hereditary component, people with a family history of diabetes are often advised to undergo regular screenings to detect early signs of the condition, especially if they have other risk factors such as being overweight, leading a sedentary lifestyle, or having high blood pressure. Identifying diabetes early through screening allows for better management and prevention of complications.",
        "20": "The ideal frequency for diabetes testing depends on an individual's risk factors and whether they have already been diagnosed with the condition. For people **without diabetes**, regular testing is generally recommended if they have risk factors such as being overweight, having a family history of diabetes, or leading a sedentary lifestyle. The American Diabetes Association suggests that adults at risk should get tested starting at age 45, and if results are normal, testing should be repeated every 3 years. For those with **prediabetes**, more frequent testing, such as annually or every 1-2 years, is recommended to monitor for progression to diabetes. For individuals who have already been **diagnosed with diabetes**, the frequency of testing depends on the type of diabetes and how well it is being managed. Those with **Type 1 diabetes** may need to monitor their blood sugar levels multiple times a day, depending on their treatment plan, especially if they are using insulin. For people with **Type 2 diabetes**, the frequency of blood sugar testing varies, but it is typically done at least once a day or several times a week, especially if they are on insulin. Regular **A1C testing** is recommended every 3 months for people whose blood sugar is not well-controlled, and every 6 months for those who have stable blood sugar control. It's important to follow the healthcare provider's guidance on the appropriate testing frequency based on individual circumstances and treatment plans.",
        "21": "Genetic factors play a significant role in the likelihood of developing diabetes, particularly Type 2 diabetes, though they also influence Type 1 diabetes to a lesser extent. In Type 2 diabetes, genetics contribute to how the body responds to insulin and how effectively it processes glucose. If a person has a parent or sibling with Type 2 diabetes, their risk of developing the condition is higher due to inherited factors that affect insulin resistance and beta-cell function (the cells in the pancreas that produce insulin). Multiple genes are thought to be involved in increasing susceptibility to Type 2 diabetes, though environmental factors such as diet, physical activity, and weight also play crucial roles in determining whether someone will actually develop the condition.",
        "22": "Obesity is a major risk factor for developing Type 2 diabetes and contributes to its onset by causing insulin resistance. Excess body fat, particularly around the abdomen, disrupts the body's ability to effectively use insulin, which is necessary for cells to absorb glucose from the bloodstream. As a result, the pancreas produces more insulin to compensate, but eventually, it cannot keep up, leading to high blood sugar levels. Obesity also causes inflammation in the body, making cells even less responsive to insulin, and leads to higher levels of fatty acids in the blood, which further impair insulin function. Additionally, excess fat affects the pancreas and liver, contributing to increased glucose production and storage. This combination of factors makes it harder for the body to regulate blood sugar levels and increases the risk of developing diabetes. Reducing weight through diet and exercise can help prevent or manage Type 2 diabetes by improving insulin sensitivity and blood sugar control.",
        "23": "Yes, there is a connection between stress and diabetes. Stress can affect blood sugar levels in several ways, particularly in people who are already at risk of or have diabetes. When a person experiences stress, the body produces stress hormones like cortisol and adrenaline, which trigger the "fight or flight" response. These hormones can increase blood sugar levels by stimulating the liver to release glucose into the bloodstream and by making the body less sensitive to insulin, a condition known as **insulin resistance**. For people with Type 2 diabetes, this can lead to higher blood sugar levels. Chronic stress can also lead to unhealthy coping behaviors, such as overeating, choosing high-sugar or high-fat foods, or being less physically active, all of which can further raise the risk of developing diabetes or worsen blood sugar control in people who already have the condition. In people with Type 1 diabetes, stress can also lead to blood sugar fluctuations, though it does so in a slightly different way. Managing stress through relaxation techniques, regular exercise, and a balanced lifestyle can help maintain better blood sugar control and overall health.",
        "24": "The pancreas plays a crucial role in the development of diabetes because it is responsible for producing insulin, a hormone that helps regulate blood sugar levels. In **Type 1 diabetes**, the immune system mistakenly attacks and destroys the insulin-producing cells in the pancreas (beta cells), leading to little or no insulin production. Without insulin, glucose cannot enter cells effectively, causing high blood sugar levels. In **Type 2 diabetes**, the pancreas initially produces enough insulin, but the body's cells become resistant to it, meaning insulin cannot be used properly to help glucose enter the cells. Over time, the pancreas struggles to produce enough insulin to overcome this resistance, and blood sugar levels rise. The pancreas also plays a role in regulating other hormones involved in glucose metabolism, and any dysfunction in these processes can contribute to the development or worsening of diabetes.",
        "25": "Response for category 2.",
        "26": "Response for category 2.",
        "27": "Response for category 2.",
        "28": "Response for category 2.",
        "29": "Response for category 2.",
        "30": "Response for category 2.",
        "31": "Response for category 2.",    
        "32": "Response for category 2.",
        "33": "Response for category 2.",
        "34": "Response for category 2.",
        "35": "Response for category 2.",
        "36": "Response for category 2.",
        "37": "Response for category 2.",
        "38": "Response for category 2.",
        "39": "Response for category 2.",
        "40": "Response for category 2.",
        "41": "Response for category 2.",
        "42": "Response for category 2.",
        "43": "Response for category 2.",    
        "44": "Response for category 2.",
        "45": "Response for category 2.",
        "46": "Response for category 2.",
        "47": "Response for category 2.",
        "48": "Response for category 2.",
        "49": "Response for category 2.",
        "50": "Response for category 2.",
        "51": "Response for category 2.",
        "52": "Response for category 2.",
        "53": "Response for category 2.",
        "54": "Response for category 2.",
        "55": "Response for category 2.",    
        "56": "Response for category 2.",
        "57": "Response for category 2.",
        "58": "Response for category 2.",
        "59": "Response for category 2.",
        "60": "Response for category 2.",
        "61": "Response for category 2.",
        "62": "Response for category 2.",
        "63": "Response for category 2.",
        "64": "Response for category 2.",
        "65": "Response for category 2.",
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
        "2": "Response for category 2.",
        "2": "Response for category 2.",
        # ... Add all 100 responses
    }

    response = responses.get(category, "Sorry, I don't understand your question.")
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
