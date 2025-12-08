import os
import boto3
import pandas as pd
import wave
import numpy as np
import zipfile
import shutil
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

# 언어별 음성 목록 (기본값은 첫 번째)
voice_options = {
    'ko-KR': [
        {'id': 'Seoyeon', 'name': 'Seoyeon (여성, Neural)', 'gender': 'Female'},
        {'id': 'InJoon', 'name': 'InJoon (남성, Standard)', 'gender': 'Male'}
    ],
    'zh-CN': [
        {'id': 'Zhiyu', 'name': 'Zhiyu (여성, Neural)', 'gender': 'Female'}
    ],
    'en-US': [
        {'id': 'Joanna', 'name': 'Joanna (여성, Neural)', 'gender': 'Female'},
        {'id': 'Matthew', 'name': 'Matthew (남성, Neural)', 'gender': 'Male'},
        {'id': 'Salli', 'name': 'Salli (여성, Neural)', 'gender': 'Female'},
        {'id': 'Kimberly', 'name': 'Kimberly (여성, Neural)', 'gender': 'Female'},
        {'id': 'Kendra', 'name': 'Kendra (여성, Neural)', 'gender': 'Female'},
        {'id': 'Justin', 'name': 'Justin (남성, Neural)', 'gender': 'Male'},
        {'id': 'Ivy', 'name': 'Ivy (여성, Neural)', 'gender': 'Female'},
        {'id': 'Joey', 'name': 'Joey (남성, Neural)', 'gender': 'Male'}
    ]
}

# 언어별 기본 음성 ID (각 언어의 첫 번째 음성)
def get_default_voice(language_code):
    """언어 코드에 따른 기본 음성 ID 반환"""
    voices = voice_options.get(language_code, voice_options['en-US'])
    return voices[0]['id'] if voices else 'Joanna'

# 언어 코드별 파일명 접두사 설정
prefix_map = {
    'ko-KR': 'KR',
    'zh-CN': 'CN',
    'en-US': 'EN'
}

@app.route('/')
def index():
    return render_template('SHVG.html')

@app.route('/api/voices/<language_code>')
def get_voices(language_code):
    """언어 코드에 따른 음성 목록 반환"""
    voices = voice_options.get(language_code, voice_options['en-US'])
    return jsonify({'voices': voices, 'default': voices[0]['id'] if voices else 'Joanna'})

def create_ssml_text(text, pitch, rate, use_neural=False):
    """텍스트를 SSML 형식으로 변환하여 피치와 속도를 적용
    Neural 엔진은 pitch를 지원하지 않으므로 pitch는 제외"""
    # SSML 특수 문자 이스케이프 처리
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    text = text.replace('"', '&quot;')
    text = text.replace("'", '&apos;')
    
    # 속도 값 포맷팅
    rate_str = f"{'+' if rate >= 0 else ''}{rate}%"
    
    # Neural 엔진은 pitch를 지원하지 않으므로 rate만 사용
    if use_neural:
        # Neural 엔진: rate만 사용
        ssml = f'<speak><prosody rate="{rate_str}">{text}</prosody></speak>'
    else:
        # Standard 엔진: pitch와 rate 모두 사용
        pitch_str = f"{'+' if pitch >= 0 else ''}{pitch}%"
        ssml = f'<speak><prosody pitch="{pitch_str}" rate="{rate_str}">{text}</prosody></speak>'
    
    return ssml

def is_neural_voice(voice_id):
    """음성이 Neural 엔진을 지원하는지 확인"""
    neural_voices = ['Seoyeon', 'Zhiyu', 'Joanna', 'Matthew', 'Salli', 'Kimberly', 
                     'Kendra', 'Justin', 'Ivy', 'Joey']
    return voice_id in neural_voices

def synthesize_to_wav(text, voice_id, pitch, rate, output_path, engine=None):
    """텍스트를 음성으로 변환하여 WAV 파일로 저장하는 헬퍼 함수
    engine: 'neural' 또는 'standard' (None이면 자동 선택)"""
    # 엔진 선택: 명시적으로 지정된 경우 그대로 사용, 아니면 자동 선택
    if engine is None:
        use_neural = is_neural_voice(voice_id)
    else:
        use_neural = (engine.lower() == 'neural')
    
    # SSML 형식으로 변환 (Neural 엔진인 경우 pitch 제외)
    ssml_text = create_ssml_text(text, pitch, rate, use_neural=use_neural)
    
    try:
        # Polly를 사용하여 음성 합성 (SSML 사용)
        synthesize_params = {
            'VoiceId': voice_id,
            'OutputFormat': 'pcm',
            'Text': ssml_text,
            'TextType': 'ssml'
        }
        
        # Neural 엔진 사용 시
        if use_neural:
            synthesize_params['Engine'] = 'neural'
        # Standard 엔진은 명시적으로 지정하지 않으면 기본값 사용
        
        response = polly_client.synthesize_speech(**synthesize_params)
        pcm_data = response['AudioStream'].read()
    except Exception as e:
        raise Exception(f'Error synthesizing speech: {str(e)}')

    # WAV 파일로 변환 및 저장
    try:
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            # 음량 최대화를 위해 PCM 데이터를 처리
            pcm_array = np.frombuffer(pcm_data, dtype=np.int16)
            if len(pcm_array) > 0:
                max_value = np.iinfo(np.int16).max
                max_abs = np.max(np.abs(pcm_array))
                if max_abs > 0:
                    pcm_array = pcm_array * (max_value / max_abs)  # 음량을 최대화
                wav_file.writeframes(pcm_array.astype(np.int16).tobytes())
    except Exception as e:
        raise Exception(f'Error creating WAV file: {str(e)}')

@app.route('/convert', methods=['POST'])
def convert():
    try:
        file = request.files['file']
        language_code = request.form['language']
        
        # 피치와 속도 값 가져오기 (기본값: 0)
        try:
            pitch = int(request.form.get('pitch', 0))
            rate = int(request.form.get('rate', 0))
        except (ValueError, TypeError):
            pitch = 0
            rate = 0
        
        # 값 범위 제한 (-50 ~ +50)
        pitch = max(-50, min(50, pitch))
        rate = max(-50, min(50, rate))
        
        if not file:
            return jsonify({'success': False, 'message': 'No file uploaded'}), 400

        # 파일 저장
        csv_file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(csv_file_path)

        # CSV 파일 읽기
        try:
            df = pd.read_csv(csv_file_path)
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error reading CSV file: {str(e)}'}), 400

        # 텍스트로 변환할 열 자동 탐색
        text_column = None
        for col in df.columns:
            if df[col].dtype == object:  # 텍스트 열을 찾기 위해 열의 데이터 유형이 문자열인지 확인
                text_column = col
                break

        if not text_column:
            return jsonify({'success': False, 'message': 'No text column found'}), 400

        # 성우 선택 가져오기 (기본값: 해당 언어의 기본 음성)
        voice_id = request.form.get('voice', get_default_voice(language_code))
        # 엔진 선택 가져오기 (기본값: Neural 지원 음성이면 'neural', 아니면 'standard')
        engine = request.form.get('engine', None)
        if engine and engine.lower() == 'auto':
            engine = None  # 'auto'는 None으로 처리하여 자동 선택
        elif engine:
            engine = engine.lower()
        prefix = prefix_map.get(language_code, 'EN')  # 기본값은 'EN'으로 설정

        # 출력 폴더 초기화
        if os.path.exists(OUTPUT_FOLDER):
            shutil.rmtree(OUTPUT_FOLDER)
        Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

        generated_files = []

        # 각 행의 텍스트를 음성으로 변환하여 WAV 파일로 저장
        for index, row in df.iterrows():
            text = str(row[text_column])  # 텍스트가 포함된 열의 데이터를 사용
            
            # 빈 텍스트 건너뛰기
            if pd.isna(text) or not text.strip():
                continue
            
            # WAV 파일 경로 생성
            file_name = os.path.join(OUTPUT_FOLDER, f"{prefix}{str(index + 1).zfill(3)}.wav")
            
            try:
                # 음성 합성 및 WAV 파일 생성
                synthesize_to_wav(text, voice_id, pitch, rate, file_name, engine=engine)
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)}), 500

            generated_files.append({'filename': file_name, 'text': text})

        if not generated_files:
            return jsonify({'success': False, 'message': 'No valid text found in CSV file'}), 400

        # 생성된 파일 정보로 CSV 파일 생성
        output_csv_path = os.path.join(OUTPUT_FOLDER, 'generated_files.csv')
        pd.DataFrame(generated_files).to_csv(output_csv_path, index=False, encoding='utf-8-sig')

        # 변환된 파일을 압축하여 다운로드 링크 제공
        zip_filename = 'output_wavs.zip'
        
        # 기존 ZIP 파일 삭제
        if os.path.exists(zip_filename):
            os.remove(zip_filename)
        
        # ZIP 파일 생성 (Windows 호환)
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(OUTPUT_FOLDER):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, OUTPUT_FOLDER)
                    zipf.write(file_path, arcname)
        
        return send_file(
            zip_filename, 
            as_attachment=True, 
            download_name='output_wavs.zip',
            mimetype='application/zip'
        )
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Unexpected error: {str(e)}'}), 500

@app.route('/test_convert', methods=['POST'])
def test_convert():
    """임의 텍스트를 음성으로 변환하는 테스트 엔드포인트"""
    try:
        text = request.form.get('testText', '').strip()
        language_code = request.form.get('testLanguage', 'ko-KR')
        
        if not text:
            return jsonify({'success': False, 'message': '텍스트를 입력해주세요.'}), 400
        
        # 피치와 속도 값 가져오기 (기본값: 0)
        try:
            pitch = int(request.form.get('testPitch', 0))
            rate = int(request.form.get('testRate', 0))
        except (ValueError, TypeError):
            pitch = 0
            rate = 0
        
        # 값 범위 제한 (-50 ~ +50)
        pitch = max(-50, min(50, pitch))
        rate = max(-50, min(50, rate))
        
        # 성우 선택 가져오기 (기본값: 해당 언어의 기본 음성)
        voice_id = request.form.get('testVoice', get_default_voice(language_code))
        # 엔진 선택 가져오기 (기본값: Neural 지원 음성이면 'neural', 아니면 'standard')
        engine = request.form.get('testEngine', None)
        if engine and engine.lower() == 'auto':
            engine = None  # 'auto'는 None으로 처리하여 자동 선택
        elif engine:
            engine = engine.lower()
        
        # 테스트용 출력 파일 경로
        test_output_path = os.path.join(OUTPUT_FOLDER, 'test_output.wav')
        Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
        
        # 음성 합성 및 WAV 파일 생성
        synthesize_to_wav(text, voice_id, pitch, rate, test_output_path, engine=engine)
        
        # WAV 파일 반환
        return send_file(test_output_path, as_attachment=False, download_name='test_audio.wav', mimetype='audio/wav')
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

if __name__ == '__main__':
    # host='0.0.0.0'으로 설정하면 모든 네트워크 인터페이스에서 접속 가능
    # port는 원하는 포트 번호로 변경 가능 (기본값: 5000)
    app.run(host='0.0.0.0', port=5000, debug=True)