import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser

# ✅ Streamlit Page Setup
st.set_page_config(page_title="AI Code Explainer + Refactor Tool", layout="wide")
st.title("💡 AI Code Explainer + Refactor Tool")
st.write("✅ Upload any code file and get instant summary, explanation, and refactoring suggestions!")

# ✅ Step 1: API Key Input
groq_api_key = st.text_input("🔑 Enter your Groq API Key:", type="password")
if not groq_api_key:
    st.warning("Please enter your API key to continue!")
    st.stop()

st.success("✅ API Key added successfully!")

# ✅ Step 2: Prompt Template for Code Analysis
prompt_template = PromptTemplate(
    template="""
You are a professional code reviewer AI.

Tasks:
1. Code Summary: Explain the code in 4-5 lines.
2. Line-by-Line Explanation: Explain each line.
3. Refactoring Suggestions: Give suggestions to optimize the code.

Code:
```{code}```

Format:
**Code Summary:**  
{{Summary}}

**Line-by-Line Explanation:**  
{{Explanation}}

**Refactoring Suggestions:**  
{{Suggestions}}
""",
    input_variables=["code"]
)

# ✅ Step 3: Groq Setup
chat = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="deepseek-r1-distill-llama-70b"
)

llm_chain = LLMChain(
    llm=chat,
    prompt=prompt_template,
    output_parser=StrOutputParser()
)

# ✅ Step 4: File Upload
uploaded_file = st.file_uploader("📂 Upload Code File", type=["py", "js", "java", "cpp", "c", "html", "css", "go", "php", "rb"])

if uploaded_file:
    code_content = uploaded_file.read().decode("utf-8")
    st.code(code_content, language="text")

    st.info("🚀 AI analyzing your code...")

    response = llm_chain.invoke({"code": code_content})
    response_text = response['text'] if isinstance(response, dict) else response

    try:
        summary = response_text.split("**Line-by-Line Explanation:**")[0].split("**Code Summary:**")[1].strip()
        explanation = response_text.split("**Line-by-Line Explanation:**")[1].split("**Refactoring Suggestions:**")[0].strip()
        suggestions = response_text.split("**Refactoring Suggestions:**")[1].strip()
    except Exception as e:
        st.error(f"⚠️ AI response parsing error: {e}")
        summary, explanation, suggestions = "Not Found", "Not Found", "Not Found"

    st.subheader("📌 Code Summary")
    st.markdown(summary)

    st.subheader("📄 Line-by-Line Explanation")
    st.markdown(explanation)

    st.subheader("🛠️ Refactoring Suggestions")
    st.markdown(suggestions)

    # ✅ Step 5: Suggest More Features Button
    if st.button("💡 Suggest How to Improve This App"):
        st.info("🚀 Getting AI suggestions on improving this app...")

        feature_prompt = PromptTemplate(
            template="""
You are a product strategist AI.

This is an AI tool that explains uploaded code files and provides refactoring suggestions using Groq + LangChain + Streamlit.

Question:
How can this app be made better, more advanced, and useful?

Give bullet-point suggestions:
- New features
- Better UI ideas
- Extra functionalities

Answer in simple words.
""",
            input_variables=[]
        )

        feature_chain = LLMChain(
            llm=chat,
            prompt=feature_prompt,
            output_parser=StrOutputParser()
        )

        feature_response = feature_chain.invoke({})
        feature_text = feature_response['text'] if isinstance(feature_response, dict) else feature_response

        st.subheader("🚀 AI Suggestions to Improve This App:")
        st.markdown(feature_text)
