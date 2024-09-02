import os
import boto3
import pandas as pd
import wave
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template
from pathlib import Path

app = Flask(__name__)

# AWS Polly 클라이언트 설정
polly_client = boto3.Session(
    aws_access_key_id='개인 엑세스 키를 넣어주세요.',
    aws_secret_access_key='개인 시크릿 키를 넣어주세요.',
    region_name='ap-northeast-2'
).client('polly')

# 업로드 및 출력 디렉토리 설정
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output_wavs'
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# 언어 코드와 Polly의 음성 ID 매핑
voice_map = {
    'ko-KR': 'Seoyeon',  # 한국어
    'zh-CN': 'Zhiyu',    # 중국어
    'en-US': 'Joanna'    # 영어
}

# 언어 코드별 파일명 접두사 설정
prefix_map = {
    'ko-KR': 'KR',
    'zh-CN': 'CN',
    'en-US': 'EN'
}

@app.route('/')
def index():
    return render_template('SHVG.html')
@app.route('/convert', methods=['POST'])
def convert():
    file = request.files['file']
    language_code = request.form['language']
    
    if not file:
        return jsonify({'success': False, 'message': 'No file uploaded'})

    # 파일 저장
    csv_file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(csv_file_path)

    # CSV 파일 읽기
    df = pd.read_csv(csv_file_path)

    # 텍스트로 변환할 열 자동 탐색
    text_column = None
    for col in df.columns:
        if df[col].dtype == object:  # 텍스트 열을 찾기 위해 열의 데이터 유형이 문자열인지 확인
            text_column = col
            break

    if not text_column:
        return jsonify({'success': False, 'message': 'No text column found'})

    # 언어에 따른 음성 ID 설정
    voice_id = voice_map.get(language_code, 'Joanna')  # 기본값은 'Joanna'로 설정
    prefix = prefix_map.get(language_code, 'EN')  # 기본값은 'EN'으로 설정

    generated_files = []

    # 각 행의 텍스트를 음성으로 변환하여 WAV 파일로 저장
    for index, row in df.iterrows():
        text = row[text_column]  # 텍스트가 포함된 열의 데이터를 사용
        response = polly_client.synthesize_speech(VoiceId=voice_id, OutputFormat='pcm', Text=text)
        pcm_data = response['AudioStream'].read()

        # WAV 파일로 변환 및 저장
        file_name = f"{OUTPUT_FOLDER}/{prefix}{str(index + 1).zfill(3)}.wav"
        with wave.open(file_name, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            # 음량 최대화를 위해 PCM 데이터를 처리
            pcm_array = np.frombuffer(pcm_data, dtype=np.int16)
            max_value = np.iinfo(np.int16).max
            pcm_array = pcm_array * (max_value / np.max(np.abs(pcm_array)))  # 음량을 최대화
            wav_file.writeframes(pcm_array.astype(np.int16).tobytes())

        generated_files.append({'filename': file_name, 'text': text})

    # 생성된 파일 정보로 CSV 파일 생성
    output_csv_path = os.path.join(OUTPUT_FOLDER, 'generated_files.csv')
    pd.DataFrame(generated_files).to_csv(output_csv_path, index=False)

    # 변환된 파일을 압축하여 다운로드 링크 제공
    zip_filename = 'output_wavs.zip'
    os.system(f"zip -r {zip_filename} {OUTPUT_FOLDER}")
    
    return send_file(zip_filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)