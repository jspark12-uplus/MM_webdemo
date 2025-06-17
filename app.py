import streamlit as st
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
import soundfile as sf
import os
import tempfile
from qwen_omni_utils import process_mm_info

# 모델 로드
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    torch_dtype="auto",
    device_map="auto"
)
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

USE_AUDIO_IN_VIDEO = True

st.title("Qwen2.5-Omni 멀티모달 인퍼런스 테스트")

# 입력 폼
with st.form("input_form"):
    user_prompt = st.text_input("프롬프트 (텍스트)", "")
    uploaded_img = st.file_uploader("이미지 업로드 (선택)", type=["jpg", "jpeg", "png"])
    uploaded_audio = st.file_uploader("오디오 업로드 (선택)", type=["wav"])
    uploaded_video = st.file_uploader("비디오 업로드 (선택)", type=["mp4"])
    submitted = st.form_submit_button("질문하기")

if submitted:
    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
                }
            ]
        },
        {
            "role": "user",
            "content": []
        }
    ]

    # 파일 저장
    with tempfile.TemporaryDirectory() as tmpdir:
        if user_prompt:
            conversation[1]["content"].append({"type": "text", "text": user_prompt})

        if uploaded_img:
            img_path = os.path.join(tmpdir, uploaded_img.name)
            with open(img_path, "wb") as f:
                f.write(uploaded_img.read())
            conversation[1]["content"].append({"type": "image", "image": img_path})

        if uploaded_audio:
            audio_path = os.path.join(tmpdir, uploaded_audio.name)
            with open(audio_path, "wb") as f:
                f.write(uploaded_audio.read())
            conversation[1]["content"].append({"type": "audio", "audio": audio_path})

        if uploaded_video:
            video_path = os.path.join(tmpdir, uploaded_video.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_video.read())
            conversation[1]["content"].append({"type": "video", "video": video_path})

        # 템플릿 적용 및 처리
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)

        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO
        )
        inputs = inputs.to(model.device).to(model.dtype)

        # 생성
        text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        decoded_text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        st.subheader("응답 텍스트")
        st.write(decoded_text[0])
