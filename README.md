# 강화학습을 통한 주식 거래 (Capstone design 2021-1)
* 산업경영공학과 16학번 김수영

## Overview
* Needs, problems
* 코로나 사태 이후 증시와 은행 예금으로 동시에 자금이 대거 유입되고 있다. 정부와 한국은행의 적극적인 유동성 공급 정책으로 시중 자금이 급증한 가운데, 이 돈이 '모(투자 위험 큰 주식) 아니면 도(안전한 예금)'로 몰리고 있는 것이다. 
 특히 최근 증시로의 민간 자금 유입은 3월 29일 45조 원(개인 투자자의 예탁금 규모)으로 역대 최대기록을 달성하고, 언론에서는 이를 '동학개미운동' 이라는 말로 대서특필할 정도로 전례 없던 규모이다.
* Goals, objectives (evaluation)
	1) Rule, 기술적 분석에 기반한 트레이딩 봇을 개발한다.
	2) 강화학습에 기반한 트레이딩 봇을 개발한다.
	3) Back-Testing을 통한 트레이딩 봇의 성능 평가 결과를 비교하여 제시한다.
	4) 강화학습을 성공적으로 수행하여, 실제 운용 가능한 수준의 트레이딩 봇을 개발한다.
## Schedule
|            Contents           | March | April |  May  | June  |   Progress   |
|-------------------------------|-------|-------|-------|-------|--------------|
|  강화학습에 대한 배경지식 확보  |   O   |       |       |       |     Link1    |
|  Nasdaq 개별 종목의 데이터 확보 |   O   |       |       |       |     Link2    |
|      데이터 셋 가공 및 구축     |   O   |       |       |       |     Link2    |
|   Rule 기반의 트레이딩 봇 개발  |       |   O   |    O    |       |     Link2    |
| 강화학습 기반의 트레이딩 봇 개발 |       |       |   O   |       |     Link2    |
| Back-Testing을 통한 봇 성능 평가|      |       |    O   |      O  |     Link2    |
|    봇 성능 개선 및 실제 운용    |     |       |       |    O    |     Link2    |

## Results
* Main code, table, graph, comparison, ...
* Web link

``` C++
void Example(int x, int y) {
   ...  
   ... // comment
   ...
}
```

## Conclusion
* Summary, contribution, ...

## Reports
* Upload or link (e.g. Google Drive files with share setting)
* Midterm: [Report](Reports/Midterm.pdf)
* Final: [Report](Reports/Final.pdf), [Demo video](Reports/Demo.mp4)
