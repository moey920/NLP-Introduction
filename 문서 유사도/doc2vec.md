# Bag of words의 장단점

- Bag of words 기반 문서 벡터의 장점
    - 벡터의 구성 요소가 직관적인 것은 bag of words 기반 기법의 큰 장점
- Bag of words 기반 문서 벡터의 단점
    - 텍스트 데이터의 양이 증가하면, 문서 벡터의 차원 증가
        - Long tail, (빈도가 낮은) 단어의 개수가 많아진다.
        - 대부분 단어의 빈도수가 0인 희소(sparse) 벡터가 생성
            - 현재 문서에 포함되지 않는 단어가 수많아진다.
            - 아주 짧은 문장이라도, 수많은 희소 벡터를 가진 벡터가 된다.
        - 문서 벡터의 차원 증가에 따른 **메모리 제약 및 비효율성 발생**
        - 문서 벡터의 차원 증가에 따른 **차원의 저주** 발생(모델링 관점에서 차원이 늘어날수록 문제가 된다.)
            - 차원이 늘어날수록 벡터 간 유사도의 의미가 사라진다.
                - 축의 관점에서 비교해야 할 축이 늘어나면서, 축 내에서 발생하는 차이가 점점 더해진다. 더해질수록 거리가 전반적으로 커진다.
                - 거리값 자체가 높고 낮음으로 유사도를 계산해야하는데, 모든 거리가 커지기때문에 유사도를 정확히 측정할 수 없어진다.

# doc2vec

> doc2vec은 문서 내 단어 간 문맥적 유사도를 기반으로 문서 벡터를 임베딩

- Input node로 문서가 추가된다. 문맥이 주어졌을 때 단어를 맞추는 문제를 풀 때 문서 벡터의 가중치로 학습되어 단어 임베딩과 비슷하게 수치형 벡터로 학습된다.(문서 + 단어 학습)
- 문서 내 단어의 임베딩 벡터를 학습하면서 문서의 임베딩 또한 지속적으로 학습
    - 특정 문장을 학습시켰을 때, 다른 문맥이 주어졌을 때 단어를 잘 맞추는 학습이 가능하다.
- 유사한 문맥의 문서 임베딩 벡터는 인접한 공간에 위치
    - 기사를 예로 들면, 날씨는 날씨끼리, 스포츠는 스포츠끼리 벡터 공간 상 인접한 공간에 위치한다.
- doc2vec은 상대적으로 **저차원의 공간**에서 문서 벡터를 생성(Bag of words의 단점을 해소한다)

### 임베딩을 통한 문장 유사도 측정 서비스

이번 실습에서는 gensim을 사용하여, doc2vec 문서 벡터를 학습하고, 이를 통해서 문서 간 유사도를 계산해 봅니다. 영화 리뷰 데이터인 IMDB dataset을 학습 데이터로 사용합니다.

```
# -*- coding: utf-8 -*-
import random
import re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from numpy import sqrt, dot

random.seed(10)

doc1 = ["homelessness has been an issue for years but never a plan to help those on the street that were once considered human who did everything from going to school work or vote for the matter"]

doc2 = ["it may have ends that do not tie together particularly well but it is still a compelling enough story to stick with"]

# 데이터를 불러오는 함수입니다.
def load_data(filepath):
    regex = re.compile('[^a-z ]')

    gensim_input = []
    with open(filepath, 'r') as f:
        for idx, line in enumerate(f):
            lowered_sent = line.rstrip().lower()
            filtered_sent = regex.sub('', lowered_sent)
            tagged_doc = TaggedDocument(filtered_sent, [idx])
            gensim_input.append(tagged_doc)
            
    return gensim_input
    
def cal_cosine_sim(v1, v2):
    # 벡터 간 코사인 유사도를 계산해 주는 함수를 완성합니다.
    top = dot(v1, v2)
    size1 = sqrt(dot(v1, v1))
    size2 = sqrt(dot(v2, v2)) 
    
    return top / (size1 * size2)
    
# doc2vec 모델을 documents 리스트를 이용해 학습하세요.
documents = load_data("text.txt")
d2v_model = Doc2Vec(window = 2, vector_size = 50)
d2v_model.build_vocab(documents) # 학습데이터 주입
d2v_model.train(documents, total_examples = d2v_model.corpus_count, epochs = 5)

# 학습된 모델을 이용해 doc1과 doc2에 들어있는 문서의 임베딩 벡터를 생성하여 각각 변수 vector1과 vector2에 저장하세요.
# 학습된 Doc2Vec 객체에 infer_vector(문장) 메소드를 사용하시면 벡터를 생성할 수 있습니다.
vector1 = d2v_model.infer_vector(doc1)
vector2 = d2v_model.infer_vector(doc2)

# vector1과 vector2의 코사인 유사도를 변수 sim에 저장하세요.
sim = cal_cosine_sim(vector1, vector2)

# 계산한 코사인 유사도를 확인합니다.
print(sim) # 0.041860808
```


