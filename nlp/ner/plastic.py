
import spacy

nlp = spacy.load("en_core_web_lg")
print("[+] Model Loaded")
text = open("plastic_hearing.txt", "r").read()
print(type(text))

print("[+] Doc loaded")
doc = nlp(text)
print("[+] nlp() done")
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
