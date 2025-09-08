"""
EEG 분석 파이프라인 체크포인트 관리 시스템
작업 상태 추적, 중간 결과 저장/로드, 파이프라인 재개 기능
"""
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
class CheckpointManager:
    """파이프라인 체크포인트 관리 클래스"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.pipeline_steps = [
            "feature_extraction",
            "ml_optimization", 
            "validation_analysis",
            "paper_results"
        ]
        
        self.status_file = self.checkpoint_dir / "pipeline_status.json"
        self.load_status()
        
    def load_status(self):
        """파이프라인 상태 로드"""
        if self.status_file.exists():
            with open(self.status_file, 'r', encoding='utf-8') as f:
                self.status = json.load(f)
        else:
            self.status = {
                "current_step": None,
                "completed_steps": [],
                "step_details": {},
                "last_updated": None,
                "session_id": None
            }
    
    def save_status(self):
        """파이프라인 상태 저장"""
        self.status["last_updated"] = datetime.now().isoformat()
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump(self.status, f, indent=2, ensure_ascii=False)
    
    def start_session(self, session_id: str = None):
        """새 세션 시작"""
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.status["session_id"] = session_id
        self.save_status()
        
    def is_step_completed(self, step: str) -> bool:
        """단계 완료 여부 확인"""
        return step in self.status["completed_steps"]
    
    def get_completed_steps(self) -> List[str]:
        """완료된 단계들 반환"""
        return self.status["completed_steps"].copy()
    
    def get_next_step(self) -> Optional[str]:
        """다음 실행할 단계 반환"""
        completed = set(self.status["completed_steps"])
        for step in self.pipeline_steps:
            if step not in completed:
                return step
        return None
    
    def save_checkpoint(self, step: str, data: Dict[str, Any], metadata: Dict[str, Any] = None):
        """체크포인트 저장"""
        timestamp = datetime.now().isoformat()
        
        checkpoint_file = self.checkpoint_dir / f"{step}_checkpoint.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
        
        metadata_file = self.checkpoint_dir / f"{step}_metadata.json"
        meta_info = {
            "step": step,
            "timestamp": timestamp,
            "session_id": self.status["session_id"],
            "data_keys": list(data.keys()),
            "metadata": metadata or {}
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(meta_info, f, indent=2, ensure_ascii=False)
        
        if step not in self.status["completed_steps"]:
            self.status["completed_steps"].append(step)
        
        self.status["current_step"] = step
        self.status["step_details"][step] = {
            "timestamp": timestamp,
            "checkpoint_file": str(checkpoint_file),
            "metadata_file": str(metadata_file),
            "data_keys": list(data.keys())
        }
        
        self.save_status()
        
    def load_checkpoint(self, step: str) -> Optional[Dict[str, Any]]:
        """체크포인트 로드"""
        checkpoint_file = self.checkpoint_dir / f"{step}_checkpoint.pkl"
        
        if not checkpoint_file.exists():
            return None
        
        try:
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            
            return data
            
        except Exception as e:
            return None
    
    def get_checkpoint_metadata(self, step: str) -> Optional[Dict[str, Any]]:
        """체크포인트 메타데이터 반환"""
        metadata_file = self.checkpoint_dir / f"{step}_metadata.json"
        
        if not metadata_file.exists():
            return None
            
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def print_status(self):
        """파이프라인 상태 출력"""
        
        if self.status["session_id"]:
            pass
        
        if self.status["last_updated"]:
            pass
        
        
        for step in self.pipeline_steps:
            if step in self.status["completed_steps"]:
                timestamp = self.status["step_details"][step]["timestamp"]
            else:
                pass
        
        next_step = self.get_next_step()
        if next_step:
            pass
        else:
            pass
    
    def resume_pipeline(self) -> Dict[str, Any]:
        """파이프라인 재개를 위한 정보 반환"""
        next_step = self.get_next_step()
        completed_steps = self.get_completed_steps()
        
        results = {}
        for step in completed_steps:
            checkpoint_data = self.load_checkpoint(step)
            if checkpoint_data:
                results[step] = checkpoint_data
        
        return {
            "next_step": next_step,
            "completed_steps": completed_steps,
            "results": results,
            "can_resume": next_step is not None or len(completed_steps) == len(self.pipeline_steps)
        }
    
    def reset_pipeline(self, confirm: bool = False):
        """파이프라인 리셋"""
        if not confirm:
            return
        
        for file in self.checkpoint_dir.glob("*_checkpoint.pkl"):
            file.unlink()
        for file in self.checkpoint_dir.glob("*_metadata.json"):
            file.unlink()
        
        self.status = {
            "current_step": None,
            "completed_steps": [],
            "step_details": {},
            "last_updated": None,
            "session_id": None
        }
        self.save_status()
        
    
    def cleanup_old_checkpoints(self, keep_days: int = 7):
        """오래된 체크포인트 정리"""
        cutoff_time = datetime.now().timestamp() - (keep_days * 24 * 3600)
        
        cleaned_files = []
        for file in self.checkpoint_dir.glob("*.pkl"):
            if file.stat().st_mtime < cutoff_time:
                file.unlink()
                cleaned_files.append(str(file))
        
        for file in self.checkpoint_dir.glob("*.json"):
            if file.name != "pipeline_status.json" and file.stat().st_mtime < cutoff_time:
                file.unlink()
                cleaned_files.append(str(file))
        
        if cleaned_files:
            pass
        
        return cleaned_files
checkpoint_manager = CheckpointManager()
def save_step_checkpoint(step: str, **kwargs):
    """단계별 체크포인트 저장 헬퍼 함수"""
    checkpoint_manager.save_checkpoint(step, kwargs)
def load_step_checkpoint(step: str):
    """단계별 체크포인트 로드 헬퍼 함수"""
    return checkpoint_manager.load_checkpoint(step)
def get_pipeline_status():
    """파이프라인 상태 조회 헬퍼 함수"""
    return checkpoint_manager.resume_pipeline()