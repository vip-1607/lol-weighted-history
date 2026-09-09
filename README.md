# LoL 가중 히스토리 모형

Bo3·Bo5에서 이전 세트 결과, 현재 연승·연패, 다음 세트 진영에 따라 A의 세트 승률이
달라지는 가상 모형을 시뮬레이션하고 과거 세트 자료로 계수를 추정합니다.

## 모형

```text
logit p(h) = logit p0 + alpha * Σ decay^(최근거리) * r
                       + gamma * signed_streak(h)
                       + beta * side_code
```

- `r`: A 승리 `+1`, 패배 `-1`
- `signed_streak`: 연승 길이는 양수, 연패 길이는 음수
- `side_code`: A가 블루면 `+0.5`, 레드면 `-0.5`, 중립이면 `0`
- `beta`: 블루와 레드 사이의 로그오즈 차이

원본 메모에서 승패 누적합의 크기 계수가 빠져 있어 `alpha`를 명시적으로 추가했습니다.
이 구현은 과거 전체를 요약한 **히스토리 의존 모형**입니다. 실제 LoL 모멘텀이 존재한다고
전제하거나 입증하지 않습니다.

## 설치와 실행

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest -q
weighted-history simulate examples/scenario.json
weighted-history fit examples/history.csv --fixed-decay 0.7 --cv-folds 3 --output fitted-model.json
weighted-history predict fitted-model.json examples/scenario.json
```

`fit`은 `logit(p0)`를 고정 offset으로 사용하고 `alpha`, `decay`, `gamma`, `beta`를
정규화 최대우도로 추정합니다. `--fixed-decay`를 생략하면 감쇠율도 추정합니다.
`--bootstrap N`은 시리즈 단위 재표집 95% 구간을, `--cv-folds K`는 시리즈 단위
교차검증 로그손실을 계산합니다.

## CSV 형식

한 행은 한 세트이며 다음 열이 필수입니다.

| 열 | 의미 |
|---|---|
| `series_id` | 시리즈 식별자 |
| `best_of` | 3 또는 5 |
| `set_number` | 1부터 연속된 세트 번호 |
| `a_won` | A 승리 1, 패배 0 |
| `p0` | 해당 시리즈 경기 전 A의 진영 중립 세트 승률 |
| `a_side` | `blue`, `red`, `neutral` |

특징은 현재 행의 결과를 보기 전에 이전 행들로만 계산합니다. 같은 시리즈에서 `p0`와
`best_of`가 바뀌거나 승부가 끝난 뒤 행이 추가되면 오류를 냅니다. 팀별 전력을 추정하는
기능은 없으므로 신뢰할 수 있는 경기 전 `p0`가 필요합니다.

주요 API는 `weighted_probability`, `simulate_series`, `build_history_features`,
`fit_weighted_history`, `predict_from_fit`입니다. 실제 데이터가 없을 때는 예제와 테스트의
합성 결과만 사용하며 실제 시장 수익성, 자동 주문, 프록시 기능은 포함하지 않습니다.
