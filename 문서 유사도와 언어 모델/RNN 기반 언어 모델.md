# RNN 기반 언어 모델

> RNN으로 문장의 각 단어가 주어졌을 때 **다음 단어를 예측**하는 문제로 언어 모델 학습

- **문자 단위 언어 모델**로 **학습 데이터 내 존재하지 않았던 단어 처리 및 생성 가능**
    - 한 문자씩 학습이 가능하다
- 모델 학습 시, 문장의 시작과 종료를 의미하는 태그(tag) 추가
    - `학습 데이터 내 문장 1 : <Start> | 포근한 | 봄 | 날씨가 | … | 입니다 | <End>`
- 문장 생성 시, 주어진 입력값부터 **순차적**으로 예측 단어 및 문자를 생성
- 고성능 언어 모델은 대용량 데이터와 이를 학습할 수 있는 하드웨어가 필수
    - OpenAI, GPT-3, BERT, transformer, attention 등(RNN 이외에도 많다)
    - 개인적으로 사용해야 할 필요가 있을 경우, 공개되어 있는 이미 학습된 모델을 사용하는 것이 좋다(전이학습)

### RNN 기반 언어 모델을 통한 간단한 챗봇 서비스 구성

이번 실습에서는 **미리 학습된 RNN 기반의 언어 모델을 불러와서** 문장 생성을 진행할 예정입니다. 본 언어 모델은 셰익스피어의 작품 내 극중 인물들의 대사로 구성되어 있는 Shakespeare 데이터셋을 사용하여 학습되었습니다.

- 학습된 언어모델의 파라미터는 checkpoints 폴더 아래에 저장되었습니다. model.load_weights()를 이용해 주어진 폴더에 저장되어 있는 데이터를 불러오세요.
    - model.load_weights() 인자로 tensorflow의 tf.train.latest_checkpoint(경로명) 함수를 사용하여, 경로 내 존재하는 파라미터 파일에서 데이터를 읽어올 수 있습니다.

- generate_text의 인자로 model과 "Juliet: "이라는 문자열을 추가하여 생성된 문장을 result 변수에 저장하세요.

```
# 경고문을 무시합니다.
import warnings
warnings.filterwarnings(action='ignore')

import pickle
import tensorflow as tf
import numpy as np


# 학습된 모델을 불러오는 함수입니다.
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        
        # 각 시점별 문자예측을 위한 LSTM 구조입니다.
        tf.keras.layers.LSTM(rnn_units,
                        # 배치단위로 학습할 때 가중치를 유지하고, 학습별로 예측을 진행하는 옵션들 
                        return_sequences=True, # 문자가 입력이 되었을 때 각 시점별로 다음 문자를 예측한다.
                        stateful=True, # 각 배치별로 동일한 LSTM weight를 유지하도록 설정
                        recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
        ])
    return model

# 학습된 모델에서 문장을 생성하는 함수입니다.
def generate_text(model, start_string):
    num_generate = 100 # 생성하는 문장이 너무 길어지지 않도록 제한하는 최대 문장 길이

    # 예측할 문자 혹은 문자열의 정수형 인덱스로 변환
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0) # 입력차원 맞추기
    text_generated = []

    model.reset_states() # 학습 마지막 시점의 모델을 불러왔기 때문에 마지막 입력이 들어갔을 떄 누적된 가중치를 초기화하여 처음부터 다시 예측할 수 있도록 한다.

    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        # 다음 발생확률이 제일 높은 문자로 예측
        predicted_id = np.argmax(predictions[-1]) # 발생할 확률이 가장 높은 문자의 인덱스값을 가져와서 
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id]) # 문장에 이어붙인다

    return (start_string + ''.join(text_generated))

# 기존 학습한 모델의 구조를 불러옵니다.
# 예측을 위해 batch_size는 1로 조절되었습니다.
model = build_model(65, 256, 1024, batch_size=1)

# model.load_weights()을 이용해 데이터를 불러오세요.
model.load_weights(tf.train.latest_checkpoint("checkpoints"))
model.build()

# char2idx, idx2char는 주어진 문자를 정수 인덱스로 매핑하는 딕셔너리 입니다.
# 자연어 처리에 딥러닝을 할 땐 문자열이 아닌 인덱스로 매핑하여 사용한다.
with open('word_index.pkl', 'rb') as f:
    char2idx, idx2char = pickle.load(f)

# "Juliet: "이라는 문자열을 추가하여 생성된 문장을 result 변수에 저장하세요.
result = generate_text(model, "Juliet: ")
print(result)
'''
Juliet: the oracle
That we remember me and my servant speak.

KING RICHARD III:
Some day to hear the truth o
'''
```
