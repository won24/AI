from transformers import pipeline

text = "바보야~"
# classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
#  한국어 버전
classifier = pipeline("sentiment-analysis", model="WhitePeak/bert-base-cased-Korean-sentiment")
result = classifier(text)
print(result)