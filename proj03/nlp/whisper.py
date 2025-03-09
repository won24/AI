# from transformers import WhisperProcessor, WhisperForConditionalGeneration
# from datasets import load_dataset

# # load model and processor
# processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
# model.config.forced_decoder_ids = None

# # load dummy dataset and read audio files
# ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
# sample = ds[0]["audio"]
# input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 


# # generate token ids
# predicted_ids = model.generate(input_features)
# # decode token ids to text
# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)

# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

# result = transcription
# print(result)


# 1. 패키지 import
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import librosa


# 2. 모델과 프로세서 로드
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.config.forced_decoder_ids = None


# 3. load m4a file
m4a_file = "6.wav"
audio_array, sampling_rate = librosa.load(m4a_file, sr=16000)  # whisper 모델은 16kHz 샘플링 레이트를 사용

# 샘플 데이터 생성
sample = {
    "array": audio_array,
    "sampling_rate": sampling_rate
}


# 4. 전처리
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 


# 5. 모델 추론
predicted_ids = model.generate(input_features)


# 6. 후처리
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription)