from pathlib import Path
DATA_DIR = Path("/Users/hyeongsuk/Desktop/workspace/Blue/DB/EEG/step3_cleaned")   # 전처리된 .set 파일 경로
SEQ_CSV  = Path("/Users/hyeongsuk/Desktop/workspace/Blue/DB/event_info2.csv")  # subject_id,S1..S6 -> Normal/BLF
FS_TARGET = 256     # 리샘플링(이미 맞춰져 있으면 None)
WIN_SEC   = 4.0
WIN_OVERLAP = 0.5
FMIN, FMAX = 1.0, 40.0
BANDS = {'delta':(1,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,40)}
EXCLUDE_APERIODIC = (8, 13)  # aperiodic 회귀에서 제외할 알파 영역
T_LIST = [30, 60, 90, 120]   # 주요 시간 포인트로 축소, 계산 효율성 개선
MDT_THRESH = 0.70                 # 또는 0.75
TIER_STRATEGY = 'roi'
TIERS = [20, 12, 8, 4]
KOOS_RESULTS_DIR = Path("/Users/hyeongsuk/Desktop/workspace/Blue/KOOS/results2")
LOG_DIR = KOOS_RESULTS_DIR / "logs"
OUTPUT_DIR = KOOS_RESULTS_DIR / "outputs"
FIG_DIR = KOOS_RESULTS_DIR / "figures"
VALIDATION_DIR = KOOS_RESULTS_DIR / "validation"
PAPER_RESULTS_DIR = KOOS_RESULTS_DIR / "paper_results"
LOG_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
FIG_DIR.mkdir(exist_ok=True, parents=True)
VALIDATION_DIR.mkdir(exist_ok=True, parents=True)
PAPER_RESULTS_DIR.mkdir(exist_ok=True, parents=True)
