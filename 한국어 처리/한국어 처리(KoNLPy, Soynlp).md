# 한국어 자연어 처리

## 자연어 처리의 기본 요소

> 자연어 처리의 기본은 단어 추출에서 시작

- 텍스트의 단어를 통해 문장의 의미, 구성 요소 및 특징을 파악 가능

```
[텍스트 1] : Hello Elice, how are you today?
[텍스트 1] : Hello | Elice(인물) | how | are | you | today(시간)
```

## 한국어에서 단어란?

- 한국어에서 단어의 기준은 명확하지 않음
    - 영어는 띄어쓰기 기준으로 문장을 나누어주면 명확하다.
    - 한국어에서 단어는 언어 단위의 기본이긴하나, 정의가 쉽지 않고 아직도 일정하게 정의내리지 못하고 있다.

- 교착어인 한국어에서 단어는 **의미적 기능**을 하는 부분과 **문법적인 기능**을 하는 부분의 조합으로 구성
    - 엘리스 + 는/은/가(조사) : 의미적 기능 + 문법적인 기능
    - 단어적인 처리를 하는 이유는 의미적 기능을 하는 부분을 추출하기 위함
    - 먹 + 다,었다,는다 : 의미를 부여하는 단어는 앞의 '먹' 뿐이다. => '먹다'
    - 한국어 자연어 처리에서는 단어의 의미적 기능과 문법적인 기능을 **구분**하는 것이 중요

# KoNLPy

## 형태소 분석

- 형태소 분석이란 주어진 한국어 텍스트를 **단어의 원형 형태로 분리해 주는 작업**
- KoNLPy는 여러 한국어 형태소 사전을 기반으로 한국어 단어를 추출해 주는 파이썬 라이브러리
    - KoNLPy를 통해 5가지 한국어 형태소 사전을 사용할 수 있다.
    - Mecab
    - 한나눔
    - 꼬꼬마(Kkma)
    - Komoran
    - Open Korean Text(Okt)
- 각 형태소 분석기 호출 방식:
```
from konlpy.tag import Kkma, Okt
kkma = Kkma()
okt = Okt()
hannanum = Hannanum()
mecab = Mecab()
komoran = Komoran(userdict=경로)
```

```
from konlpy.tag import Kkma

sent = "안녕 나는 엘리스야 반가워. 너의 이름은 뭐야?"

kkma = Kkma()
print(kkma.nouns(sent)) # ['안녕', '나', '엘리스', '너', '이름', '뭐']
print(kkma.pos(sent)) # [('안녕', 'NNG'), ('나', 'NP'), ('는', 'JX’), ('엘리스', 'NNG'), ('야', 'JX’), ...

print(kkma.sentences(sent)) # ['안녕 나는 엘리스야 반가워. 너의 이름은 뭐야?']
```

```
from konlpy.tag import Okt

sent = "안녕 나는 엘리스야 반가워. 너의 이름은 뭐야?"

okt = Okt()
print(okt.nouns(sent)) # ['안녕', '나', '엘리스', '너', '이름', '뭐']
print(okt.pos(sent)) # [('안녕', 'Noun'), ('나', 'Noun'), ('는', 'Josa’), ('엘리스', 'Noun’), ...

print(okt.pos(sent, stem = True)) # ... ('반갑다', 'Adjective’) ...
```

> 각 형태소 사전별 형태소 표기 방법 및 기준의 차이가 존재

# soynlp

> 사전 기반의 단어 처리의 경우, 미등록 단어 문제가 발생할 수 있음

- soynlp는 학습 데이터 내 자주 발생하는 패턴을 기반으로 단어의 경계선을 구분

```
[문장 1]: 보코하람 테러로 소말리아에서 전쟁이 있었어요
꼬꼬마 기준 추출된 명사: [보, 보코, 코, 테러, 소말리, 전쟁]
OKT 기준 추출된 명사: [보코하람, 테러, 소말리아, 전쟁]
```

- 단어는 연속으로 등장하는 글자의 조합이며 글자 간 연관성이 높다는 가정
- 한국어의 어절은 좌 – 우 구조로 2등분 할 수 있다(의미적 기능 문법적 기능)

```
from soynlp.utils import DoublespaceLineCorpus
from soynlp.word import WordExtractor
from soynlp.noun import LRNounExtractor_v2

train_data = DoublespaceLineCorpus(학습데이터의 경로) # 데이터 기반 패턴 학습

noun_extractor = LRNounExtractor_v2()
nouns = noun_extractor.train_extract(train_data) # [할리우드, 모바일게임 ...

word_extractor = WordExtractor()
words = word_extractor.train_extract(train_data) # [클린턴, 트럼프, 프로그램 ...
```

### soynlp를 통한 한국어 전처리

soynlp는 한국어 단어 추출 중 발생할 수 있는 미등록 단어 문제를 해결할 수 있는 전처리 라이브러리입니다. soynlp는 학습 데이터에서 자주 발생하는 패턴을 기반으로 단어의 경계선을 구분하여 단어를 추출합니다.

이번 실습에서는 신문 기사를 학습 데이터로 사용하여 명사 목록을 학습한 뒤, 주어진 문장에서 명사를 추출할 예정입니다.

```
from soynlp.utils import DoublespaceLineCorpus
from soynlp.noun import LRNounExtractor_v2

sent = '트와이스 아이오아이 좋아여 tt가 저번에 1위 했었죠?'

# 학습에 사용할 데이터가 train_data에 저장되어 있습니다.
corpus_path = 'articles.txt'
train_data = DoublespaceLineCorpus(corpus_path)
print("학습 문서의 개수: %d" %(len(train_data)))

# LRNounExtractor_v2 객체를 이용해 train_data에서 명사로 추정되는 단어를 nouns 변수에 저장하세요.
noun_extr = LRNounExtractor_v2()
nouns = noun_extr.train_extract(train_data) # 명사가 학습데이터에서 추출된다.

# 생성된 명사의 개수를 확인해봅니다.
print(len(nouns))

# 생성된 명사 목록을 사용해서 sent에 주어진 문장에서 명사를 sent_nouns 리스트에 저장하세요.
sent_nouns = []
for word in sent.split() :
    if word in nouns :
        sent_nouns.append(word)

print(sent_nouns)
# ['트와이스', '아이오아이', '1위']
```
