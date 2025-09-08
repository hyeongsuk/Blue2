import os
import sys
import json
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import traceback
import re

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eeg_optimization_log.txt'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 설정 및 경로 import
from config import (
    DATA_DIR, SEQ_CSV, FS_TARGET, WIN_SEC, WIN_OVERLAP, 
    BANDS, EXCLUDE_APERIODIC, T_LIST, MDT_THRESH, 
    OUTPUT_DIR, FIG_DIR
)

# 유틸리티 임포트
from utils_io import list_set_files, read_seq_map, subject_from_filename, load_raw_set, lens_of
from windowing import slice_passages
from features_band_aperiodic import extract_all_features
from time_to_detection import auc_vs_time_band, compute_mdt, bootstrap_ci_subject, auc_vs_time_enhanced
from channel_selection import pick_by_roi
from plots import plot_auc_vs_time
from ensemble_optimization import run_ensemble_optimization_integrated

def create_real_eeg_labels(seq_df):
    """실제 EEG 데이터에서 레이블 생성"""
    logger.debug(f"Creating labels. DataFrame columns: {seq_df.columns}")
    logger.debug(f"First few rows:\n{seq_df.head()}")
    
    labels = {}
    for _, row in seq_df.iterrows():
        try:
            subject_id = str(row['Subjects'])
            sequence = row['Sequence']
            
            if sequence == 1:
                labels[f"{subject_id}_S1"] = 0
                labels[f"{subject_id}_S2"] = 0
                labels[f"{subject_id}_S3"] = 0
                labels[f"{subject_id}_S4"] = 1
                labels[f"{subject_id}_S5"] = 1
                labels[f"{subject_id}_S6"] = 1
            elif sequence == 2:
                labels[f"{subject_id}_S1"] = 1
                labels[f"{subject_id}_S2"] = 1
                labels[f"{subject_id}_S3"] = 1
                labels[f"{subject_id}_S4"] = 0
                labels[f"{subject_id}_S5"] = 0
                labels[f"{subject_id}_S6"] = 0
        except Exception as e:
            logger.error(f"Error processing row: {row}")
            logger.error(f"Error details: {e}")
    
    logger.debug(f"Created {len(labels)} labels")
    return labels

def extract_memory_efficient_features(file_path, labels_map):
    """메모리 효율적인 특징 추출"""
    features = []
    labels = []
    
    try:
        # 메모리 효율적인 로딩
        raw = load_raw_set(file_path, preload=False)
        raw.pick_eeg_channels()
        
        # 윈도우 슬라이딩
        win_list = slice_passages(raw, win_sec=WIN_SEC, overlap=WIN_OVERLAP)
        
        for w in win_list:
            # 기본 특징 추출
            feats_dict = extract_all_features(
                w['data'], 
                raw.info['sfreq'], 
                BANDS, 
                EXCLUDE_APERIODIC, 
                WIN_SEC, 
                WIN_OVERLAP, 
                raw.ch_names
            )
            
            # 피쳐 병합
            window_features = []
            for band in BANDS.keys():
                window_features.append(np.mean(feats_dict['traditional_band_powers'][band]))
                window_features.append(np.mean(feats_dict['relative_band_powers'][band]))
            
            # 레이블 매핑
            filename = os.path.basename(file_path)
            match = re.match(r'(\d+)_.*?_(\d+)_', filename)
            if match:
                subject_id, passage_num = match.groups()
                passage = f"S{passage_num}"
                label = labels_map.get(f"{subject_id}_{passage}")
                
                if label is not None:
                    features.append(window_features)
                    labels.append(label)
    
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        logger.error(traceback.format_exc())
    
    return features, labels

def run_real_eeg_optimization():
    """EEG 데이터 최적화 파이프라인"""
    logger.info("Starting EEG optimization pipeline")
    
    # 데이터 준비
    files = list_set_files(DATA_DIR)
    logger.debug(f"Found {len(files)} files")
    
    seq_map = read_seq_map(SEQ_CSV)
    logger.debug(f"Sequence map shape: {seq_map.shape}")
    
    labels_map = create_real_eeg_labels(seq_map)
    
    # 특징 추출
    all_features = []
    all_labels = []
    
    for file_path in files:
        features, labels = extract_memory_efficient_features(file_path, labels_map)
        all_features.extend(features)
        all_labels.extend(labels)
    
    # NumPy 배열로 변환
    X = np.array(all_features)
    y = np.array(all_labels)
    
    logger.debug(f"Total features shape: {X.shape}")
    logger.debug(f"Total labels shape: {y.shape}")
    
    # 모델 최적화
    optimization_results = run_ensemble_optimization_integrated(X, y)
    
    # AUC 곡선 생성
    curve_baseline = auc_vs_time_band(pd.DataFrame({
        'feat': list(X), 
        'y': y
    }), T_LIST)
    
    # 결과 저장
    curve_baseline.to_csv(OUTPUT_DIR / "baseline_auc_curve.csv", index=False)
    
    logger.info("EEG optimization pipeline completed successfully")
    return optimization_results

if __name__ == "__main__":
    try:
        results = run_real_eeg_optimization()
        print("Optimization completed successfully.")
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}")
        logger.critical(traceback.format_exc())
        sys.exit(1)