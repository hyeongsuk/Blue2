"""
파이프라인 상태 확인 스크립트
체크포인트 상태 조회, 재개 정보 제공
"""
import argparse
from pathlib import Path
from checkpoint_manager import checkpoint_manager, get_pipeline_status
def main():
    parser = argparse.ArgumentParser(description='EEG 분석 파이프라인 상태 확인')
    parser.add_argument('--reset', action='store_true', help='파이프라인 리셋 (주의: 모든 체크포인트 삭제)')
    parser.add_argument('--cleanup', type=int, default=7, help='N일 이전 체크포인트 정리 (기본값: 7일)')
    parser.add_argument('--detail', action='store_true', help='상세 정보 표시')
    
    args = parser.parse_args()
    
    if args.reset:
        confirm = input("⚠️  모든 체크포인트를 삭제하시겠습니까? (yes/no): ")
        if confirm.lower() == 'yes':
            checkpoint_manager.reset_pipeline(confirm=True)
        else:
        return
    
    if args.cleanup:
        checkpoint_manager.cleanup_old_checkpoints(keep_days=args.cleanup)
    
    checkpoint_manager.print_status()
    
    pipeline_info = get_pipeline_status()
    
    if pipeline_info["next_step"]:
        if pipeline_info["next_step"] == "feature_extraction":
        elif pipeline_info["next_step"] == "ml_optimization":
        elif pipeline_info["next_step"] == "validation_analysis":
        elif pipeline_info["next_step"] == "paper_results":
    else:
    
    if args.detail:
        
        for step in checkpoint_manager.pipeline_steps:
            if checkpoint_manager.is_step_completed(step):
                metadata = checkpoint_manager.get_checkpoint_metadata(step)
                if metadata:
                    if metadata.get('metadata'):
        
        output_dir = Path("../results/outputs")
        if output_dir.exists():
            for file in output_dir.glob("*.csv"):
                size = file.stat().st_size / (1024*1024)  # MB
if __name__ == "__main__":
    main()