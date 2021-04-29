# N-gram 기반 언어 모델

> **언어 모델**이란? 주어진 문장이 텍스트 데이터에서 발생할 확률을 계산하는 모델

- 얼마나 실제로 발생가능한 문장인지 확률을 계산한다.
```
문장 1 : 포근한 | 봄 | 날씨가 | 이어질 | 것으로 | 전망됩니다.

텍스트 데이터, P(문장) = 0.233
```

- 언어 모델을 통해 자동 문장 생성이 가능
    - 포털에서 검색 시 가장 적합한 문장을 자동완성해주는 기능
    - toyscript(https://toyscript.azurewebsites.net/) 웹 서비스에서 영화 타이틀 자동 완성 기능으로 사용해보기!

- 챗봇 내 핵심 요소 중 하나(확률값 기반 문장 생성)
- 문장의 발생 확률은 **단어가 발생할 조건부 확률의 곱**으로 계산
    ```
    문장 1 : 포근한 | 봄 | 날씨가 | 이어질 | 것으로 | 전망됩니다.

    P(문장) = P(포근한) × P(봄|포근한) × P(날씨가|포근한, 봄) × P(이어질|포근한, 봄, 날씨가) × … × P(전망됩니다|포근한, 봄, 날씨가, 이어질, 것으로)
    ```
- **N-gram**을 사용하여 **단어의 조건부 확률을 근사**
    - 오래된 단어보다는 최근 단어가 현재 단어를 생성할 확률에 큰 영향을 미친다는 가정
    - 계산량이 줄어들고 정확 근사도가 향상된다.
    ```
    문장 1 : 포근한 | 봄 | 날씨가 | 이어질 | 것으로 | 전망됩니다.

    [Tri-gram 기준= P(문장) ≈ P(날씨가|포근한, 봄) × P(이어질|봄, 날씨가) × … × P(전망됩니다|이어질, 것으로)
    ```
    - 각 N-gram 기반 조건부 확률은 데이터 내 **각 n-gram의 빈도수**로 계산
    ```
    P(날씨가|포근한, 봄) = 전체 데이터 내 "포근한 봄 날씨가" 의 빈도수 / 전체 데이터에서 "포근한 봄"의 빈도수
    ```
- 문장 생성 시, 주어진 단어 기준 **최대 조건부 확률**의 단어를 다음 단어로 생성
    ```
    생성되는 문장 : 무더운 | 여름 | ?

    P(엘리스 | 여름) = 0.02
    P(여름 | 바다) = 0.5
    P(날씨 | 무더운, 여름) = 0.87
    따라서 다음 단어로 '날씨'를 선택하여 '무더운 여름 날씨' 가 생성된다.
    …

### N-gram 언어 모델

이번 실습에서는 변수 data에 주어진 문장을 사용해서 간단한 bi-gram 기반 언어 모델을 직접 만들어 볼 예정입니다.

```
data = ['this is a dog', 'this is a cat', 'this is my horse','my name is elice', 'my name is hank']

def count_unigram(docs):
    unigram_counter = dict()
    # docs에서 발생하는 모든 unigram의 빈도수를 딕셔너리 unigram_counter에 저장하여 반환하세요.
    for doc in docs :
        for word in doc.split() :
            if word not in unigram_counter :
                unigram_counter[word] = 1
            else :
                unigram_counter[word] += 1
    return unigram_counter

def count_bigram(docs):
    bigram_counter = dict()
    # docs에서 발생하는 모든 bigram의 빈도수를 딕셔너리 bigram_counter에 저장하여 반환하세요.
    for doc in docs :
        # zip(리스트 1, 리스트 2) 함수를 사용하면, [((리스트 1의 원소), (리스트 2의 원소))]와 같이 두 리스트를 하나의 리스트로 만들 수 있습니다.
        words = doc.split()
        for word1, word2 in zip(words, words[1:]) :
            if (word1, word2) not in bigram_counter :
                bigram_counter[(word1, word2)] = 1
            else :
                bigram_counter[(word1, word2)] += 1
    
    return bigram_counter

# 입력되는 문장의 발생 확률을 계산, 예측할 문장, unigram의 빈도수 딕셔너리, bigram의 빈도수 딕셔너리를 인자로 받습니다.
def cal_prob(sent, unigram_counter, bigram_counter):
    words = sent.split()
    result = 1.0
    # sent의 발생 확률을 계산하여 변수 result에 저장 후 반환하세요.
    for word1, word2 in zip(words, words[1:]) :
        top = bigram_counter[(word1, word2)]
        bottom = unigram_counter[word1]
        result *= float(top/bottom)
    
    return result

# 주어진data를 이용해 unigram 빈도수, bigram 빈도수를 구하고 "this is elice" 문장의 발생 확률을 계산해봅니다.
unigram_counter = count_unigram(data)
bigram_counter = count_bigram(data)
print(cal_prob("this is elice", unigram_counter, bigram_counter))
```
