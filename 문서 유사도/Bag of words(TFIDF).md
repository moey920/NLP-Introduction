# 문서 유사도 측정

- 문서는 다양한 요소와 이들의 상호작용으로 구성
    - 단어 - 형태소 - 문장 - 문단
    - 가장 기본 단위인 단어 조차 문서와 관련된 다양한 정보를 포함
        - 형태소, 키워드, 개체명(Named entity : 사람, 조직 등), 중의적 단어 등
    - 상위 개념인 문장 또한 추가적인 정보를 제공
        - 목적어, 주어, 문장 간 관계, 상호참조해결(여러 문장 사이에서 발생하는 객체가 무엇인지) 등 
    - 그렇기 때문에 문서의 가장 기본 단위인 **단어를 활용하여 문서를 표현**
    - 문서 유사도를 측정하기 위해 단어 기준으로 생성한 문서 벡터 간의 코사인 유사도를 사용
        - 정확한 문서 유사도 측정을 위해 문서의 특징을 잘 보존하는 벡터 표현 방식이 중요

# Bag of words

> 문서 내 **단어의 빈도수를 기준**으로 문서 **벡터를 생성**

- 직관적이고, 널리 사용된다.
- 자주 발생하는 단어가 문서의 특징을 나타낸다는 것을 가정
- Bag of words 문서 벡터의 차원은 데이터 내 발생하는 모든 단어의 개수와 동일
- Bag of words 문서 벡터는 합성어를 독립적인 단어로 개별 처리
    - log off / log is => log / off / is : 로그오프를 구분하지 못하고 통나무로 해석된다.

```
[문서 1] : these | are | five | IT | companies …
[문서 2] : these | five | great | singers …
```

||these |are |five |**IT** |companies |great |**singers** |a|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|문서1 |3 |7 |1 |**6** |1 |0 |0 |0|
|문서2 |2 |4 |1 |0 |0 |1 |**4** |0|

# Bag of N-grams

> N-gram은 연속된 N개의 단어를 기준으로 텍스트 분석을 수행

- Bag of N-grams은 **n-gram의 발생 빈도를 기준**으로 문서 벡터를 표현
    - 여러 n-gram도 사용할 수 있다.
- 자주 발생하는 단어가 문서의 주요 내용 및 특징을 항상 효과적으로 표현하지는 않음
    - 그리고, 그러나, 잘, 오늘, 만약 등

```
N = 1 (unigram) : 포근한 | 봄 | 날씨가 | 이어질 | 것으로 | 전망되며 …
N = 2 (bi-gram) : 포근한 봄 | 봄 날씨가 | 날씨가 이어질 | 이어질 것으로 | …
N = 3 (tri-gram) : 포근한 봄 날씨가 | 봄 날씨가 이어질 | 날씨가 이어질 것으로 | …
```

## TF-IDF((term frequency– inverse document frequency)

- 문서 내 상대적으로 자주 발생하는 단어가 더 중요하다는 점을 반영
- 단어 자체가 모든 데이터에서 많이 발생했다면 점수를 낮추어 문서 간 유사도를 반영할 때 큰 영향을 주지 못하도록 방지한다.
    - 단어의 상대적인 중요성(특정 문서에서의 중요성)을 높힐 수 있다.

```
문서 1에서 단어 "봄"의 TF-IDF 점수 = 
문서 1 내 "봄"의 빈도수 / 문서1 내 모든 단어의 빈도수 
× log(데이터 내 총 문서의 개수 / 데이터 내 "봄"이 들어간 문서의 개수)
```

### Bag of words 기반 문서 벡터 생성

이번 실습에서는 scikit-learn의 CountVectorizer를 사용하여, bag of words 문서 벡터를 만들어 보는 실습을 진행할 예정입니다. 영화 리뷰 데이터인 text.txt에 저장되어 있는 IMDB dataset을 사용하여 각 리뷰별 문서 벡터를 만들어 보세요.
- CountVectorizer : bag of words 벡터

```
import re
from sklearn.feature_extraction.text import CountVectorizer

regex = re.compile('[^a-z ]')

with open("text.txt", 'r') as f:
    documents = []
    for line in f:
        # doucments 리스트에 리뷰 데이터를 저장하세요.
        filtered_doc = regex.sub("", line.rstrip())
        documents.append(filtered_doc)
        
# CountVectorizer() 객체를 이용해 Bag of words 문서 벡터를 생성하여 변수 X에 저장하세요.  
cv = CountVectorizer()
X = cv.fit_transform(documents)

# 변수 X의 차원을 변수 dim에 저장하세요.
dim = X.shape
# X 변수의 차원을 확인해봅니다.
print(dim) # (454, 12640)

# 위에서 생성한 CountVectorizer() 객체에서 첫 10개의 칼럼이 의미하는 단어를 words_feature 변수에 저장하세요.
words_feature = cv.get_feature_names()[:10]

# CountVectorizer() 객체의 첫 10개 칼럼이 의미하는 단어를 확인해봅니다.
print(words_feature) # ['aal', 'aba', 'abandon', 'abandoned', 'abbot', 'abducted', 'abets', 'abilities', 'ability', 'abilitytalent']

# 단어 "comedy"를 의미하는 칼럼의 인덱스 값을 idx 변수에 저장하세요.
idx = cv.vocabulary_['comedy']
# 단어 "comedy"의 인덱스를 확인합니다.
print(idx) # 2129

# 첫 번째 문서의 Bag of words 벡터를 vec1 변수에 저장하세요.
vec1 = X[0]
# 첫 번째 문서의 Bag of words 벡터를 확인합니다.
print(vec1)
'''
  (0, 9686)	4
  (0, 5525)	4
  (0, 6010)	4
  (0, 1761)	1
  (0, 2129)	1
  (0, 9081)	1
  ...
'''
```

### TF-IDF Bag of words 기반 문서 벡터 생성

이번 실습에서는 scikit-learn의 TfidfVectorizer를 사용하여, TF-IDF 기반 bag of words 문서 벡터를 만들어 보는 실습을 진행할 예정입니다. TfidfVectorizer의 사용법은 CountVectorizer의 사용법과 동일합니다.

TfidfVectorizer(ngram_range=(1, 2))으로 객체를 생성하고 fit_transform() 메소드를 사용하여 TF-IDF 기반 Bag of N-grams 문서 행렬을 생성하세요.

- ngram_range=(1, 2)는 데이터 내 unigram과 bigram을 사용하여 문서 벡터를 생성한다는 의미를 갖고 있습니다.

```
import re
from sklearn.feature_extraction.text import TfidfVectorizer

regex = re.compile('[^a-z ]')

# 리뷰 데이터를 가져옵니다
with open("text.txt", 'r') as f:
    documents = []
    for line in f:
        lowered_sent = line.rstrip().lower()
        filtered_sent = regex.sub('', lowered_sent)
        documents.append(filtered_sent)

# TfidfVectorizer() 객체를 이용해 Bag of words 문서 벡터를 생성하여 변수 X에 저장하세요.
tv = TfidfVectorizer()
X = tv.fit_transform(documents)

# 변수 X의 차원을 변수 dim1에 저장하세요.
dim1 = X.shape
# X 변수의 차원을 확인해봅니다.
print(dim1) # (454, 12136)

# 첫 번째 문서의 Bag of words를 vec1 변수에 저장하세요.
vec1 = X[0]
# 첫 번째 문서의 Bag of words를 확인합니다.
print(vec1)
'''
  (0, 5679)	0.058640619958889736
  (0, 8003)	0.10821800789540346
  (0, 11827)	0.03351360629176965
  ...
'''

# 위에서 생성한 TfidfVectorizer() 객체를 이용해 TF-IDF 기반 Bag of N-grams 문서 벡터를 생성하세요.
unibi_v = TfidfVectorizer(ngram_range = (1,2))
unibigram_X = unibi_v.fit_transform(documents)


# 생성한 TF-IDF 기반 Bag of N-grams 문서 벡터의 차원을 변수 dim2에 저장하세요.
dim2 = unibigram_X.shape
# 문서 벡터의 차원을 확인합니다.
print(dim2) # (454, 74358)
```

### Bag of words 기반 문서 유사도 측정 서비스

이번 실습에서는 앞서 학습한 TF-IDF 기반 Bag of words 모델을 사용하여 주어진 문서의 유사도를 코사인 유사도를 사용하여 계산할 예정입니다.

```
# 경고문을 제거합니다.
import warnings
warnings.filterwarnings(action='ignore')

import pickle
from sklearn.metrics.pairwise import cosine_similarity

sent1 = ["I first saw this movie when I was a little kid and fell in love with it at once."]
sent2 = ["Despite having 6 different directors, this fantasy hangs together remarkably well."]

with open('bow_models.pkl', 'rb') as f:
    # 저장된 모델을 불러와 객체와 벡터를 각각vectorizer와 X에 저장하세요.
    # 객체 : 학습된 TF-IDF 기반 Bag of words의 TfidfVectorizer()객체 / 벡터(X) : 문서 벡터(학습된 문서 행렬)
    vectorizer, X = pickle.load(f)

# sent1, sent2 문장을 vectorizer 객체의 transform() 함수를 이용해 변수 vec1, vec2에 저장합니다.
vec1 = vectorizer.transform(sent1)
vec2 = vectorizer.transform(sent2)

#  vec1과 vec2의 코사인 유사도를 변수 sim1에 저장합니다.
sim1 = cosine_similarity(vec1, vec2)
# 두 벡터의 코사인 유사도를 확인해봅니다.
print(sim1) # [[0.00936629]]

# vec1과 행렬 X의 첫 번째 문서 벡터 간 코사인 유사도를 변수 sim2에 저장합니다.
sim2 = cosine_similarity(vec1, X[0])
# X의 첫 번째 문서와 vec1의 코사인 유사도를 확인해봅니다.
print(sim2) # [[0.04116601]]
```
