from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# 모델과 토크나이저 로드
model_name = "circulus/kobart-trans-ko-en-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# 번역할 한국어 문장
text = "안녕하세요. 오늘 날씨가 정말 좋네요!"
# 토큰화 및 입력 변환 (token_type_ids 제거)
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
# 모델 추론 수행 (token_type_ids 제거)
outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
# 번역 결과 디코딩
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("번역 결과:", translated_text)