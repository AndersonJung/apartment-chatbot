"""
반달마을 건영아파트 관리규약 챗봇
==================================
  pip install streamlit google-genai
  set GEMINI_API_KEY=AQ.your_key
  streamlit run chatbot.py

같은 폴더에 규약_전문.md 와 조문_인덱스.json 이 있어야 합니다.
"""

import os
import re
import json
import pathlib

import streamlit as st
from google import genai
from google.genai import types

# 모델 폴백 체인. RECITATION 으로 막히면 다음 모델로 넘어간다.
MODELS = ["gemini-3.1-flash-lite", "gemini-3-flash-preview", "gemini-3.5-flash"]

# 규약집에서 빼기로 한 별표들. 목차에서 확보한 제목·페이지를 안내에 쓴다.
BYEOLPYO = {
    1: ("관리대상물", "제4조", 52),
    2: ("전유부분의 범위", "제5조제1항", 53),
    3: ("공용부분의 범위", "제5조제2항", 53),
    4: ("관리비의 세대별 부담액 산정방법", "제72조제1항", 54),
    5: ("공동 사용료 등의 산정방법", "제73조제1항", 54),
    6: ("사용료 등의 산정방법", "제73조제2항", 55),
    7: ("기타 이용료 등의 산정방법", "제73조제2항", 55),
}


@st.cache_data
def load():
    text = pathlib.Path("규약_전문.md").read_text(encoding="utf-8")
    idx = json.loads(pathlib.Path("조문_인덱스.json").read_text(encoding="utf-8"))
    return text, {a["no"]: a for a in idx}


def build_system_prompt(full_text: str) -> str:
    guide = "\n".join(
        f"- [별표{n}] {title} ({ref} 관련) — 관리규약집 {page}페이지"
        for n, (title, ref, page) in BYEOLPYO.items()
    )
    return f"""당신은 '반달마을 건영아파트' 입주민을 돕는 관리규약 안내 도우미입니다.

# 답변 규칙
1. 아래 <관리규약> 원문에 근거해서만 답합니다. 규약에 없는 내용은 절대 지어내지 않습니다.
2. 답변 마지막 줄에 반드시 근거 조항을 이 형식으로 적습니다:
   근거: 제○○조 제○항
   여러 개면 쉼표로 나열합니다. 근거를 못 찾으면 답하지 마십시오.
3. **조문을 그대로 길게 복사하지 마십시오.** 반드시 쉬운 말로 요약해서 설명합니다.
   입주민이 알아들을 수 있는 일상적인 표현을 씁니다.
4. 규약에 명시되지 않은 사항은 이렇게 답합니다:
   "관리규약에 명시되어 있지 않습니다. 관리사무소에 문의해 주세요."
5. 법적 해석이나 분쟁 판단이 필요한 질문은 답을 단정하지 말고,
   해당 조문의 취지만 설명한 뒤 관리사무소·입주자대표회의 문의를 안내합니다.

# 별표 안내 규칙 (중요)
아래 별표들은 원문이 주어지지 않았습니다. 내용을 절대 추측하지 마십시오.
질문의 답이 이 별표에 있으면, 제목과 페이지를 알려주고 확인을 안내합니다.

{guide}

예시 답변:
"관리비 세대별 부담액 산정방법은 [별표4] '관리비의 세대별 부담액 산정방법'에
규정되어 있습니다. 관리규약집 54페이지를 확인하시거나 관리사무소에 문의해 주세요.
근거: 제72조 제1항"

# 관리규약 전문
<관리규약>
{full_text}
</관리규약>
"""


def ask(client, history, system):
    """모델 폴백 체인. RECITATION/빈 응답이면 다음 모델로."""
    contents = [
        types.Content(role=m["role"], parts=[types.Part.from_text(text=m["content"])])
        for m in history
    ]
    last_reason = ""
    for model in MODELS:
        try:
            r = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.2,
                    max_output_tokens=8192,
                ),
            )
            text = (r.text or "").strip()
            if text:
                return text, None
            last_reason = str(r.candidates[0].finish_reason)
        except Exception as e:
            last_reason = str(e)
            if "429" in last_reason or "RESOURCE_EXHAUSTED" in last_reason:
                return None, "요청이 몰렸습니다. 30초 뒤 다시 시도해 주세요."
    return None, f"답변 생성에 실패했습니다. ({last_reason})"


def cited_articles(answer: str):
    """맨 끝 '근거:' 줄에 적힌 조문만 원문 열람 대상으로 삼는다.
    본문 중의 '법 제14조', '영 제11조' 같은 상위법령 인용은 제외."""
    m = re.search(r"근거\s*[:：](.+)$", answer, re.S | re.M)
    if not m:
        return []
    return sorted({int(n) for n in re.findall(r"제\s*(\d+)\s*조", m.group(1))})


# ---------------------------------------------------------------------------
st.set_page_config(page_title="반달마을 건영아파트 관리규약 안내", page_icon="🏠")
st.title("🏠 관리규약 안내 도우미")
st.caption("반달마을 건영아파트 · 2025.07.25 개정 관리규약 기준")

st.warning(
    "⚠️ **동·호수, 이름, 이웃과의 분쟁 내용 등 개인정보는 입력하지 마세요.** "
    "입력 내용이 AI 학습에 사용될 수 있습니다."
)
st.caption(
    "본 서비스는 입주민 편의를 위한 **비공식 참고 도구**입니다. "
    "공식 해석은 관리사무소 및 입주자대표회의를 따릅니다. "
    "관리비 산정표 등 **별표는 수록되어 있지 않으므로** 규약집 원본을 확인해 주세요."
)

if not os.environ.get("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
    st.stop()

# --- 비밀번호 잠금 ---
APP_PASSWORD = st.secrets.get("APP_PASSWORD") or os.environ.get("APP_PASSWORD")

if APP_PASSWORD:
    if not st.session_state.get("authed"):
        st.info("입주민 전용입니다. 관리사무소에서 안내받은 비밀번호를 입력해 주세요.")
        pw = st.text_input("비밀번호", type="password")
        if pw:
            if pw == APP_PASSWORD:
                st.session_state.authed = True
                st.rerun()
            else:
                st.error("비밀번호가 올바르지 않습니다.")
        st.stop()

full_text, index = load()
system = build_system_prompt(full_text)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

with st.sidebar:
    st.metric("수록 조문", f"{len(index)}개")
    st.metric("규약 전문", f"{len(full_text):,}자")
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.msgs = []
        st.rerun()
    st.divider()
    st.caption(
        "⚠️ 본 답변은 참고용입니다. 공식 해석은 관리사무소 및 "
        "입주자대표회의를 따릅니다. 별표(관리비 산정표 등)는 수록되어 있지 "
        "않으므로 해당 내용은 규약집 원본을 확인해 주세요."
    )

if "msgs" not in st.session_state:
    st.session_state.msgs = []

if not st.session_state.msgs:
    st.info(
        "관리규약에 대해 물어보세요.\n\n"
        "예시\n"
        "- 동별 대표자가 되려면 어떤 자격이 필요한가요?\n"
        "- 관리비를 연체하면 어떻게 되나요?\n"
        "- 반려동물을 키우려면 동의를 받아야 하나요?\n"
        "- 층간소음 분쟁은 어떻게 처리되나요?"
    )

for m in st.session_state.msgs:
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.markdown(m["content"])

if q := st.chat_input("궁금한 내용을 입력하세요"):
    st.session_state.msgs.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("관리규약을 확인하는 중..."):
            answer, err = ask(client, st.session_state.msgs, system)

        if err:
            st.warning(err)
        else:
            st.markdown(answer)
            st.session_state.msgs.append({"role": "model", "content": answer})

            # 원문은 모델이 아니라 코드가 보여준다 (RECITATION 회피 + 정확성 보장)
            for no in cited_articles(answer):
                if no in index:
                    a = index[no]
                    with st.expander(f"📄 제{no}조 【{a['title']}】 원문 보기"):
                        st.text(a["body"])
