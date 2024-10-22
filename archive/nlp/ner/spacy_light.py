

import spacy

nlp = spacy.load("en_core_web_lg")

sentences = [
    "Drivers download the Rootmobile app and take a test drive that typically lasts two or three weeks.",
    "Root provides a quote that rewards good driving behavior and allows customers to switch their insurance policy.",
    "Customers can purchase and manage their policy through the app."
]

for text in sentences:
    print("-------------------------------------------------------------")
    doc = nlp(str(text))
    for token in doc:
        print(token.text, '-->', token.pos_, token.dep_)

    print("----------")
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)
