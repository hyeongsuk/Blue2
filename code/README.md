
# KOOS — EEG Time-to-Detection Pipeline (MNE)
# Access Check Completed

이 폴더는 전처리 완료된 EEGLAB `.set` 파일(자동 ICA 기반 아티팩트 제거, 1–40 Hz 필터 적용)을 입력으로
**Blue-filter vs. Normal** 구분의 *시간-의존 디코딩*을 수행하고, **MDT(최소 검출시간)**, **AUC-vs-Time 곡선**, 
**채널 축소(20→12→8→4)**, **등가성(짧은 vs 120초)**, **부분구간 재표본화(강건성)**, 
그리고 논문용 Figure를 자동 생성합니다.

## 요구사항
- Python 3.9+
- `requirements.txt` 설치: `pip install -r requirements.txt`

## 입력
- 전처리된 EEGLAB 파일 폴더: `DATA_DIR` (config.py에서 설정)
- 렌즈 순서 매핑 CSV: `SEQ_CSV` — 열: `subject_id,S1,S2,S3,S4,S5,S6` 값은 `Normal` 또는 `BLF`

## 빠른 시작
1. `config.py`에서 `DATA_DIR`, `SEQ_CSV`, `FS_TARGET`, `TIER_STRATEGY` 등을 설정
2. 실행: `python run_all.py`
3. 결과: `outputs/` (곡선/표), `figs/` (그림), `logs/` (로그)

## 구조
- `config.py`                : 경로/파라미터/채널 티어 설정
- `utils_io.py`              : 파일 로드, 매핑 로드, 채널 핸들링
- `windowing.py`             : S*_Start/S*_End 기반 윈도 분할(4s, 50% overlap)
- `features_band_aperiodic.py`: 밴드파워(δ–γ) + aperiodic(1/f slope/offset)
- `features_riemann.py`      : 공분산(OAS)→탄젠트공간 임베딩
- `time_to_detection.py`     : AUC-vs-Time, LOSO, MDT, 부트스트랩 CI, 음성대조군
- `channel_selection.py`     : ROI 기반/데이터 기반 채널 축소(20→12→8→4)
- `equivalence.py`           : T vs 120초 AUC 등가성(부트스트랩 90% CI, Δ=0.03)
- `resampling_subsegments.py`: 무작위 부분구간 재표본화(예: 30초) 강건성
- `plots.py`                 : AUC-vs-Time, 중요도, 요약 그림
- `run_all.py`               : 전체 파이프라인 오케스트레이션

## 주의
- 모든 교차검증은 **피험자 단위 분리(LOSO)**로 데이터 누출 방지
- 윈도 overlap로 인해 같은 지문의 윈도는 항상 같은 fold에만 존재
- 채널 이름은 10–20 규칙(O/P/CP/C 등)을 권장, 일부 누락 시 자동 대체 시도
