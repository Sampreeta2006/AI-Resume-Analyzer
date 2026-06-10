import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import io


# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI Career Recommender",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 AI-Powered Resume Analyzer & Career Recommender")
st.write("Upload your resume and get AI-powered career recommendations, ATS score, skill gap analysis, and career roadmap using O*NET occupational data.")

# =====================================
# CACHE AI MODEL
# =====================================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
def extract_resume_skills(text, skills_list):

    found = []

    text = text.lower()

    for skill in skills_list:

        if skill.lower() in text:
            found.append(skill)

    return list(set(found))

model = load_model()

# =====================================
# FILE UPLOAD
# =====================================

uploaded_file = st.file_uploader(
    "Upload Resume",
    type=["pdf", "docx"]
)

if uploaded_file is not None:

    resume_text = ""

    # =====================================
    # PDF EXTRACTION
    # =====================================

    if uploaded_file.name.endswith(".pdf"):

        pdf_reader = PdfReader(uploaded_file)

        for page in pdf_reader.pages:

            text = page.extract_text()

            if text:
                resume_text += text

    # =====================================
    # DOCX EXTRACTION
    # =====================================

    elif uploaded_file.name.endswith(".docx"):

        doc = Document(uploaded_file)

        for para in doc.paragraphs:
            resume_text += para.text + "\n"

    st.success("Resume Uploaded Successfully")

    st.subheader("📄 Extracted Resume Text")
    st.text_area(
        "Resume",
        resume_text,
        height=250
    )

    # =====================================
    # RESUME SKILL DETECTION
    # =====================================

    skills_master = pd.read_csv(
        r"data/skills_master.csv"
    )

    skills_list = skills_master["Skill"].tolist()

    resume_skills = extract_resume_skills(
        resume_text,
        skills_list
    )
    st.subheader("🛠 Detected Resume Skills")

    if resume_skills:

        for skill in resume_skills:
            st.success(skill)

    else:

        st.warning(
            "No skills detected from skills_master.csv"
        )

    # =====================================
    # LOAD O*NET DATA
    # =====================================
    # =====================================
    # LOAD O*NET DATA
    # =====================================

    file_path = r"data/ONET/db_30_3_excel/Occupation Data.xlsx"

    df = pd.read_excel(
        file_path,
        engine="openpyxl"
    )

    df["occupation_text"] = (
        df["Title"].fillna("")
        + " "
        + df["Description"].fillna("")
    )

    # =====================================
    # AI ANALYSIS
    # =====================================

    with st.spinner("Analyzing Resume..."):

        resume_embedding = model.encode(
            resume_text,
            convert_to_tensor=False
        )

        occupation_embeddings = model.encode(
            df["occupation_text"].tolist(),
            convert_to_tensor=False
        )

        scores = cosine_similarity(
            [resume_embedding],
            occupation_embeddings
        )[0]

        df["score"] = scores

    # =====================================
    # TOP MATCHES
    # =====================================

    top_matches = (
        df.sort_values(
            by="score",
            ascending=False
        )
        .head(10)
    )

    st.subheader("🎯 Top 10 Career Matches")

    for i, (_, row) in enumerate(
        top_matches.iterrows(),
        start=1
    ):

        st.success(
            f"{i}. {row['Title']} ({row['score']*100:.2f}%)"
        )

    # =====================================
    # BEST MATCH
    # =====================================

    best_match = top_matches.iloc[0]

    st.subheader("🏆 Best Career Match")

    st.info(
        f"""
Role: {best_match['Title']}

Match Score: {best_match['score']*100:.2f}%
"""
    )

    st.subheader("📖 Role Description")

    st.write(
        best_match["Description"]
    )

     # =====================================
    # REQUIRED SKILLS
    # =====================================

    skills_df = pd.read_excel(
        r"data/ONET/db_30_3_excel/Essential Skills.xlsx",
        engine="openpyxl"
    )

    skills_df["O*NET-SOC Code"] = (
        skills_df["O*NET-SOC Code"].astype(str)
    )

    best_code = str(best_match["O*NET-SOC Code"])
    base_code = best_code[:7]

    occupation_skills = skills_df[
        skills_df["O*NET-SOC Code"].str.startswith(base_code, na=False)
    ]

    occupation_skills = occupation_skills[
        occupation_skills["Scale Name"] == "Importance"
    ]

    top_skills = (
        occupation_skills
        .sort_values("Data Value", ascending=False)
        .head(10)
    )

    required_skills = list(
     dict.fromkeys(
        top_skills["Element Name"].tolist()
     )
    )
    # MISSING SKILLS ANALYSIS
    # =====================================

    missing_skills = []

    resume_skills_lower = [
        skill.lower()
        for skill in resume_skills
    ]

    for skill in required_skills:

        if skill.lower() not in resume_skills_lower:
            missing_skills.append(skill)
    missing_skills = list(
        dict.fromkeys(missing_skills)
    )        
    st.subheader("🎯 Top Required Skills")

    for skill in required_skills:
        st.success(skill)

    st.subheader("❌ Missing Skills")

    if missing_skills:

        for skill in missing_skills:
            st.error(skill)

    else:

        st.success(
            "No missing skills detected"
        )
        # =====================================
    # SKILL GAP RECOMMENDATIONS
    # =====================================

    st.subheader("🎓 Skill Gap Recommendations")

    for skill in missing_skills:

        if skill == "Critical Thinking":
            st.info("Practice business case studies and problem-solving exercises.")

        elif skill == "Active Learning":
            st.info("Complete advanced courses in analytics and AI.")

        elif skill == "Reading Comprehension":
            st.info("Read technical blogs, research papers, and documentation.")

        elif skill == "Writing":
            st.info("Improve report writing and documentation skills.")

        elif skill == "Speaking":
            st.info("Practice presentations and communication skills.")

        elif skill == "Active Listening":
            st.info("Improve teamwork and stakeholder communication.")
        required_skills = [
        str(skill)
        for skill in required_skills
        if pd.notna(skill)
    ]

    missing_skills = [
        str(skill)
        for skill in missing_skills
        if pd.notna(skill)
    ]
    # =====================================
    # SKILLS ANALYSIS CHART
    # =====================================

        # =====================================
    # SKILLS ANALYSIS CHART
    # =====================================

    st.subheader("📈 Skills Analysis")

    matched_count = len(required_skills) - len(missing_skills)
    missing_count = len(missing_skills)

    # Prevent invalid values
    if matched_count < 0:
        matched_count = 0

    if missing_count < 0:
        missing_count = 0

    total_skills = matched_count + missing_count

    if total_skills > 0:

        fig, ax = plt.subplots(figsize=(2,2))

        ax.pie(
            [matched_count, missing_count],
            labels=["Matched", "Missing"],
            autopct="%1.1f%%"
        )

        ax.set_title("Skills Match Analysis")

        st.pyplot(fig)

    else:

        st.warning(
            "Not enough skill data available for chart generation."
        )

    # =====================================
    # KNOWLEDGE AREAS
    # =====================================

    knowledge_df = pd.read_excel(
        r"data/ONET/db_30_3_excel/Knowledge.xlsx",
        engine="openpyxl"
    )

    knowledge_df["O*NET-SOC Code"] = (
        knowledge_df["O*NET-SOC Code"].astype(str)
    )

    occupation_knowledge = knowledge_df[
        knowledge_df["O*NET-SOC Code"].str.startswith(base_code, na=False)
    ]

    occupation_knowledge = occupation_knowledge[
        occupation_knowledge["Scale Name"] == "Importance"
    ]

    top_knowledge = (
        occupation_knowledge
        .sort_values("Data Value", ascending=False)
        .head(5)
    )

    st.subheader("📚 Important Knowledge Areas")

    for knowledge in top_knowledge["Element Name"].tolist():
        st.info(knowledge)

    # =====================================
    # CAREER ROADMAP
    # =====================================

    st.subheader("🛣️ Career Roadmap")

    roadmap = []

    for skill in required_skills:

        if "Programming" in skill:
            roadmap.append("Learn Advanced Python and Build Real Projects")

        elif "Mathematics" in skill:
            roadmap.append("Improve Statistics and Mathematics")

        elif "Critical Thinking" in skill:
            roadmap.append("Practice Data Analysis Case Studies")

        elif "Active Learning" in skill:
            roadmap.append("Complete Advanced Online Courses")

    roadmap = list(dict.fromkeys(roadmap))

    for step in roadmap[:5]:
        st.warning(step)

    # =====================================
    # ATS SCORE
    # =====================================

        # =====================================
    # ATS SCORE
    # =====================================
    occupation_score = best_match["score"] * 100

    matched_skills = len(required_skills) - len(missing_skills)

    skill_score = (
    matched_skills /
    max(len(required_skills), 1)
    ) * 100
    ats_score = (
    occupation_score * 0.3 +
    skill_score * 0.7
    )

    ats_score = ats_score + 40

    ats_score = min(ats_score, 100)
    st.subheader("📊 ATS Score")
    if ats_score >= 80:
        st.success(f"ATS Score: {ats_score:.2f}%")
    elif ats_score >= 60:
        st.warning(f"ATS Score: {ats_score:.2f}%")
    else:
        st.error(f"ATS Score: {ats_score:.2f}%")

    st.progress(min(int(ats_score), 100))

    # =====================================
    # DOWNLOAD REPORT
    # =====================================

    report = f"""
AI CAREER RECOMMENDER REPORT

Best Career Match:
{best_match['Title']}

ATS Score:
{ats_score:.2f}%

Top Skills:
{', '.join(required_skills)}

Missing Skills:
{', '.join(missing_skills)}

Knowledge Areas:
{', '.join(top_knowledge['Element Name'].tolist())}
"""

    st.download_button(
        label="📥 Download Report",
        data=report,
        file_name="career_report.txt",
        mime="text/plain"
    )

    # =====================================
    # SUMMARY
    # =====================================

    st.subheader("📊 Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Occupations Checked",
            len(df)
        )

    with col2:
        st.metric(
            "Missing Skills",
            len(missing_skills)
        )

    with col3:
        st.metric(
            "ATS Score",
            f"{ats_score:.2f}%"
        )
