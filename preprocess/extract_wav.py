import os
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
from threading import Lock

def extract_audio_from_mp4(mp4_path, output_dir, sample_rate=16000):
    """
    MP4 파일에서 WAV 오디오 추출
    
    Args:
        mp4_path: MP4 파일 경로
        output_dir: 출력 디렉토리
        sample_rate: 샘플링 레이트 (기본: 16kHz)
    """
    mp4_file = Path(mp4_path)
    output_file = Path(output_dir) / f"{mp4_file.stem}.wav"
    
    # 출력 파일이 이미 존재하면 스킵
    if output_file.exists():
        return f"스킵: {output_file.name} (이미 존재)"
    
    try:
        # FFmpeg 명령어로 오디오 추출
        cmd = [
            'ffmpeg',
            '-i', str(mp4_path),
            '-vn',  # 비디오 스트림 제외
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            '-ar', str(sample_rate),  # 샘플링 레이트
            '-ac', '1',  # 모노 채널
            '-y',  # 덮어쓰기
            str(output_file)
        ]
        
        # subprocess로 실행 (출력 숨기기)
        result = subprocess.run(cmd, 
                              capture_output=True, 
                              text=True, 
                              check=True)
        
        return f"완료: {output_file.name}"
        
    except subprocess.CalledProcessError as e:
        return f"오류: {mp4_file.name} - {e.stderr}"
    except Exception as e:
        return f"오류: {mp4_file.name} - {str(e)}"

def process_videos_batch(input_dir, output_dir, sample_rate=16000, max_workers=4):
    """
    배치로 비디오 파일들의 오디오 추출
    
    Args:
        input_dir: MP4 파일들이 있는 디렉토리
        output_dir: WAV 파일들을 저장할 디렉토리
        sample_rate: 샘플링 레이트
        max_workers: 동시 처리할 스레드 수
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 출력 디렉토리 생성
    output_path.mkdir(parents=True, exist_ok=True)
    
    # MP4 파일 목록 가져오기
    mp4_files = list(input_path.glob("*.mp4"))
    
    if not mp4_files:
        print("MP4 파일을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(mp4_files)}개의 MP4 파일을 찾았습니다.")
    print(f"출력 디렉토리: {output_path}")
    print(f"샘플링 레이트: {sample_rate}Hz")
    print(f"동시 처리 스레드: {max_workers}")
    print("-" * 50)
    
    # 진행 상황 추적용
    progress_lock = Lock()
    success_count = 0
    error_count = 0
    
    def update_progress(result):
        nonlocal success_count, error_count
        with progress_lock:
            if "완료:" in result:
                success_count += 1
            elif "오류:" in result:
                error_count += 1
            print(result)
    
    # 멀티스레딩으로 처리
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 작업 제출
        futures = []
        for mp4_file in mp4_files:
            future = executor.submit(extract_audio_from_mp4, mp4_file, output_path, sample_rate)
            futures.append(future)
        
        # 결과 처리
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(futures), 
                          desc="오디오 추출 진행"):
            result = future.result()
            update_progress(result)
    
    print("-" * 50)
    print(f"처리 완료!")
    print(f"성공: {success_count}개")
    print(f"오류: {error_count}개")
    print(f"스킵: {len(mp4_files) - success_count - error_count}개")

def parse_arguments():
    """
    Command line arguments 파싱
    """
    parser = argparse.ArgumentParser(
        description="MP4 파일에서 WAV 오디오를 배치로 추출합니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "input_dir",
        type=str,
        help="MP4 파일들이 있는 입력 디렉토리 경로"
    )
    
    parser.add_argument(
        "output_dir", 
        type=str,
        help="WAV 파일들을 저장할 출력 디렉토리 경로"
    )
    
    parser.add_argument(
        "--sample-rate", "-sr",
        type=int,
        default=16000,
        help="오디오 샘플링 레이트 (Hz)"
    )
    
    parser.add_argument(
        "--max-workers", "-w",
        type=int,
        default=4,
        help="동시 처리할 스레드 수"
    )
    
    return parser.parse_args()

# 메인 실행
if __name__ == "__main__":
    # Command line arguments 파싱
    args = parse_arguments()
    
    # FFmpeg 설치 확인
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      capture_output=True, 
                      check=True)
        print("FFmpeg가 설치되어 있습니다.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("오류: FFmpeg가 설치되지 않았습니다.")
        print("다음 명령어로 설치하세요:")
        print("Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg")
        print("CentOS/RHEL: sudo yum install ffmpeg")
        print("macOS: brew install ffmpeg")
        exit(1)
    
    # 입력 디렉토리 확인
    if not os.path.exists(args.input_dir):
        print(f"오류: 입력 디렉토리가 존재하지 않습니다: {args.input_dir}")
        exit(1)
    
    # 오디오 추출 실행
    process_videos_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        max_workers=args.max_workers
    )