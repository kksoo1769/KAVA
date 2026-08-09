# 토크나이저 비교 예시

현재 프론티어 LLM들에서 한국어를 사용하면 영어보다 토큰 사용량이 약 1.5배 정도 높다. 따라서 한국어 중심의 LLM은 비용과 속도 면에서 반드시 한국어에서 토큰 효율적인 토크나이저가 필요하다.

## 비교 tokenizers
`|`: 토큰 구분자
- EXAONE-4.0: `LGAI-EXAONE/EXAONE-4.0-1.2B`
- Qwen3: `Qwen/Qwen3-8B`
- KANANA-1.5: `kakaocorp/kanana-1.5-2.1b-base`
- Llama: `unsloth/Llama-3.2-1B`

### 1. 한국어

원문:

```text
게임 오버(Game Over, 게임 끝)는 비디오 게임을 진행하다가 스테이지를 모두 클리어 후 엔딩이 끝나거나 진행 도중 플레이어의 패배로 게임이 더 이상 진행되지 않는 경우에 쓰는 말이다.
```

EXAONE-4.0 `(47)`:

```text
게임| 오버|(|Game| Over|,| 게임| 끝|)|는| 비디오| 게임|을| 진행|하|다가| 스테이지|를| 모두| 클리어| 후| 엔딩|이| 끝나|거나| 진행| 도중| 플레이어|의| 패배|로| 게임|이| 더| 이상| 진행|되|지| 않|는| 경우|에| 쓰|는| 말|이다|.
```

Qwen3 `(71)`:

```text
게|임| 오|버|(Game| Over|,| 게|임| |끝|)|는| 비|디|오| 게|임|을| 진행|하다|가| |스|테|이|지를| 모두| 클|리|어| 후| |엔|딩|이| |끝|나|거나| 진행| 도|중| |플|레이|어|의| |패|배|로| 게|임|이| 더| 이상| 진행|되지| 않는| 경우|에| |쓰|는| 말|이다|.
```

KANANA-1.5 `(54)`:

```text
게임| 오|버|(Game| Over|,| 게임| 끝|)는| 비|디오| 게임|을| 진행|하다|가| |스|테|이지|를| 모두| 클|리어| 후| 엔|딩|이| 끝|나|거나| 진행| 도|중| 플레이|어|의| 패|배|로| 게임|이| 더| 이상| 진행|되지| 않는| 경우|에| 쓰|는| 말|이다|.
```

Llama-3.2 `(54)`:

```text
게임| 오|버|(Game| Over|,| 게임| 끝|)는| 비|디오| 게임|을| 진행|하다|가| |스|테|이지|를| 모두| 클|리어| 후| 엔|딩|이| 끝|나|거나| 진행| 도|중| 플레이|어|의| 패|배|로| 게임|이| 더| 이상| 진행|되지| 않는| 경우|에| 쓰|는| 말|이다|.
```

### 2. 한국어

원문:

```text
비둘기의 꽁지깃은 보통 12장이지만, 이 품종은 20∼30개나 되고 이것을 부채꼴로 펴는 것이 공작과 닮았다.
```

EXAONE-4.0 `(43)`:

```text
비|둘기|의| 꽁|지|깃|은| 보통| |1|2|장이|지만|,| 이| 품종|은| |2|0|∼|3|0|개|나| 되|고| 이것|을| 부채|꼴|로| 펴|는| 것|이| 공작|과| 닮|았|다|.
```

Qwen3 `(53)`:

```text
비|둘|기|의| |꽁|지|깃|은| 보|통| |1|2|장|이|지만|,| 이| |품|종|은| |2|0|∼|3|0|개|나| 되|고| 이것|을| 부|채|꼴|로| |펴|는| 것이| 공|작|과| |닮|았다|.
```

KANANA-1.5 `(47)`:

```text
비|둘|기의| |꽁|지|깃|은| 보|통| |12|장이|지만|,| 이| 품|종|은| |20|∼|30|개|나| 되|고| 이것|을| 부|채|꼴|로| |펴|는| 것이| 공|작|과| |닮|았다|.
```

Llama-3.2 `(47)`:

```text
비|둘|기의| |꽁|지|깃|은| 보|통| |12|장이|지만|,| 이| 품|종|은| |20|∼|30|개|나| 되|고| 이것|을| 부|채|꼴|로| |펴|는| 것이| 공|작|과| |닮|았다|.
```

### 3. 한국어

원문:

```text
정보통신망법 변경에 따라 질문에 대한 답변 알림을 지속적으로 받으시려면, 10월 24일 이후 꼭 새로운 휴대전화 신규 인증을 받으셔야 합니다.
```

EXAONE-4.0 `(42)`:

```text
정|보통|신|망|법| 변경|에| 따라| 질문|에| 대한| 답변| 알림|을| 지속|적으로| 받|으|시|려면|,| |1|0|월| |2|4|일| 이후| 꼭| 새로운| 휴대|전화| 신규| 인증|을| 받|으|셔야| 합니다|.
```

Qwen3 `(53)`:

```text
정보|통신|망|법| 변경|에| 따라| 질문|에| 대한| 답변| 알|림|을| 지|속|적으로| 받|으|시|려|면|,| |1|0|월| |2|4|일| 이후| |꼭| 새로운| |휴|대|전|화| 신|규| 인|증|을| 받|으|셔|야| 합니다|.
```

KANANA-1.5 `(45)`:

```text
정보|통신|망|법| 변경|에| 따라| 질문|에| 대한| 답변| 알|림|을| 지|속|적으로| 받|으|시|려|면|,| |10|월| |24|일| 이후| 꼭| 새로운| 휴|대|전|화| 신규| 인증|을| 받|으|셔|야| 합니다|.
```

Llama-3.2 `(45)`:

```text
정보|통신|망|법| 변경|에| 따라| 질문|에| 대한| 답변| 알|림|을| 지|속|적으로| 받|으|시|려|면|,| |10|월| |24|일| 이후| 꼭| 새로운| 휴|대|전|화| 신규| 인증|을| 받|으|셔|야| 합니다|.
```

### 4. 영어

원문:

```text
The Independent Jane For all the love, romance and scandal in Jane Austen’s books, what they are really about is freedom and independence.
```

EXAONE-4.0 `(28)`:

```text
The| Independent| Jane| For| all| the| love|,| romance| and| scandal| in| Jane| Austen|’|s| books|,| what| they| are| really| about| is| freedom| and| independence|.
```

Qwen3 `(28)`:

```text
The| Independent| Jane| For| all| the| love|,| romance| and| scandal| in| Jane| Aust|en|’s| books|,| what| they| are| really| about| is| freedom| and| independence|.
```

KANANA-1.5 `(28)`:

```text
The| Independent| Jane| For| all| the| love|,| romance| and| scandal| in| Jane| Aust|en|’s| books|,| what| they| are| really| about| is| freedom| and| independence|.
```

Llama-3.2 `(28)`:

```text
The| Independent| Jane| For| all| the| love|,| romance| and| scandal| in| Jane| Aust|en|’s| books|,| what| they| are| really| about| is| freedom| and| independence|.
```

### 5. 영어

원문:

```text
Collins offer of marriage showed an independence seldom seen in heroines of the day.
```

EXAONE-4.0 `(16)`:

```text
Collins| offer| of| marriage| showed| an| independence| seldom| seen| in| heroin|es| of| the| day|.
```

Qwen3 `(17)`:

```text
Coll|ins| offer| of| marriage| showed| an| independence| seldom| seen| in| hero|ines| of| the| day|.
```

KANANA-1.5 `(17)`:

```text
Coll|ins| offer| of| marriage| showed| an| independence| seldom| seen| in| hero|ines| of| the| day|.
```

Llama-3.2 `(17)`:

```text
Coll|ins| offer| of| marriage| showed| an| independence| seldom| seen| in| hero|ines| of| the| day|.
```

## 결론: EXAONE-4.0 선택 이유

위 예시들을 종합하면 한국어 중심 학습에는 **EXAONE-4.0** 토크나이저가 가장 적합하다.

### 1. 한국어 토큰 효율이 가장 높다

동일한 한국어 문장을 가장 적은 토큰으로 표현한다. 한국어 예시 3개의 토큰 수 합계는 다음과 같다.

| Tokenizer | 한국어 예시 합 (1+2+3) | EXAONE 대비 |
| --- | --- | --- |
| **EXAONE-4.0** | **132** | — |
| KANANA-1.5 | 146 | +10.6% |
| Llama-3.2 | 146 | +10.6% |
| Qwen3 | 177 | +34.1% |

### 2. 단어 또는 형태소 경계를 잘 보존한다

EXAONE은 `게임`, `비디오`, `플레이어`, `휴대`처럼 의미 단위를 통째로 유지하는 반면, Qwen3는 `게|임`, `비|디|오`처럼 음절 단위로 과분할되는 경향이 강하다. 또한, KAKANA와 Llama는 같은 tokenizer를 사용하는 것으로 보이며 마찬가지로 한국어에서 주로 음절 단위로 분할한다. 의미 단위가 보존될수록 모델이 토큰 시퀀스로부터 의미를 학습하기 쉽고, 표현 낭비가 줄어들 수 있다.

### 3. 영어 성능도 뒤지지 않는다

영어 예시(4, 5)에서는 네 토크나이저가 28 / 16~17 토큰 수준으로 비슷하며, 오히려 EXAONE이 `Austen`을 분할 없이 유지(예시 4)하고 예시 5에서 16토큰으로 가장 적다. 즉 한국어 효율을 얻는 대가로 영어를 희생하지 않는다.

### 4. 숫자를 단일 수로 분할한다

EXAONE과 Qwen3은 `1|2`, `2|0`처럼 숫자를 한 자리씩 분리하지만 KANANA와 Llama는 `12`, `20`으로 묶는다. 한 자리씩 분리하는 방식이 수치 연산의 일관성 측면에서 더 나은 일반화를 가진다.

### 종합

한국어 압축률, 의미 단위 보존, 영어 호환성을 모두 만족하므로 EXAONE-4.0 토크나이저를 채택한다.
