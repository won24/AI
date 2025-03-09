# from transformers import pipeline

# # text = "The Golden State Warriors are an American professional basketball team based in San Francisco."
# text = ""

# # classifier = pipeline("ner", model="stevhliu/my_awesome_wnut_model")
# classifier = pipeline("ner", model="KPF/KPF-bert-ner")
# result = classifier(text)
# print(result)


from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


tokenizer = AutoTokenizer.from_pretrained("Leo97/KoELECTRA-small-v3-modu-ner")
model = AutoModelForTokenClassification.from_pretrained("Leo97/KoELECTRA-small-v3-modu-ner")
ner = pipeline("ner", model=model, tokenizer=tokenizer)
example = "서울시 관악구 봉천로 4길은 우리집"
ner_results = ner(example)
print(ner_results)