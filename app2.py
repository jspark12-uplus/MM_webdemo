import streamlit as st
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
import torch
import soundfile as sf
import os
import tempfile
from qwen_omni_utils import process_mm_info

# Model load status flag
@st.cache_resource(show_spinner=True)
def try_load_model():
    try:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-Omni-7B",
            torch_dtype="bfloat16",
            # device_map="auto",
            attn_implementation="flash_attention_2",
            cache_dir="/data/public/models",
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(
            "Qwen/Qwen2.5-Omni-7B", cache_dir="/data/public/models",)

        return model, processor, True
    except Exception as e:
        return None, None, False

model, processor, is_model_ready = try_load_model()

#st.set_page_config(page_title="Qwen2.5-Omni Demo", layout="centered")
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎥 Qwen2.5-Omni 멀티모달 데모")

# 상태 표시
if is_model_ready:
    st.success("🟢 모델 로딩 완료")
else:
    with st.empty():
        import time
        for i in range(20):
            st.info(f"🕐 모델 로딩 중... {i*5}%", icon="🕐")
            time.sleep(0.05)
    st.warning("🚧 아직 모델이 준비되지 않았습니다. 입력은 가능하지만 응답 버튼은 비활성화됩니다.")

st.caption("텍스트, 이미지, 오디오, 비디오 중 원하는 입력을 조합해서 AI에게 질문해보세요.")

with st.form("input_form"):
    user_prompt = st.text_input("📝 프롬프트 입력", "")
    uploaded_img = st.file_uploader("🖼️ 이미지 업로드 (선택)", type=["jpg", "jpeg", "png"])
    uploaded_audio = st.file_uploader("🔊 오디오 업로드 (선택)", type=["wav"])
    uploaded_video = st.file_uploader("🎞️ 비디오 업로드 (선택)", type=["mp4"])
    submitted = st.form_submit_button("질문하기", disabled=not is_model_ready)

if submitted and is_model_ready:
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ]
        },
        {
            "role": "user",
            "content": []
        }
    ]

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

        st.info("⏳ 모델 응답 생성 중...", icon="⏳")

        # Inference
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)

        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True
        )
        inputs = {k: v.to(model.device, dtype=model.dtype) if v.dtype.is_floating_point else v.to(model.device) for k, v in inputs.items()}

        text_ids, audio_tensor = model.generate(**inputs, use_audio_in_video=True)
        decoded_text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        st.success("✅ 응답 완료!")
        st.markdown("### 💬 모델 응답")
        st.write(decoded_text[0])

        # 오디오 출력
        audio_path = os.path.join(tmpdir, "output.wav")
        sf.write(audio_path, audio_tensor.reshape(-1).detach().cpu().numpy(), samplerate=24000)
        st.audio(audio_path, format="audio/wav")
