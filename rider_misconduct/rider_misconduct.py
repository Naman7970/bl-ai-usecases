import streamlit as st
import pandas as pd
import re
import asyncio
import openai
from io import BytesIO
import pdb

# pdb.set_trace()
import logging

logging.basicConfig(
    level=logging.DEBUG,  # or INFO
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

logger.info("This is an info log")
logger.debug("This is a debug log")

# Set your OpenAI key here
openai.api_key = st.secrets["OPENAI_API_KEY"]

client = openai.AsyncOpenAI(api_key=openai.api_key)

st.set_page_config(page_title="Misconduct Score Dashboard", layout="centered")
st.title("🚨 Rider Misconduct Scorer")

uploaded_file = st.file_uploader("Upload a CSV (must have a 'feedback' column)", type="csv")


# Better regex: only matches 0-10
def extract_score(text: str):
    match = re.search(r"\b(10|[0-9])\b", text)
    return int(match.group()) if match else None


async def get_score_async(feedback):
    try:
        logger.info("This is an info log inside score async")
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are tasked with analyzing customer feedback regarding delivery partner behavior.\n"
                        "Your goal is to classify each comment into one of five categories, based on the severity of the behavior described.\n" 
                        "Here are the categories:\n"
                        "1. Normal: The comment describes a neutral or positive interaction, with no issues reported.\n"
                        "Examples:\n"
                        "- The rider was very polite and friendly throughout the delivery.\n"
                        "- Great service! The rider greeted me with a smile.\n"
                        "- The rider was very professional and even helped with directions.\n"
                        "\n\n"
                        "2. Mild: The comment mentions minor issues or inconveniences, but overall describes an acceptable experience.\n"
                        "Examples:\n" 
                        "- The rider seemed in a rush and didn't explain what was happening much.\n"
                        "- The rider was a bit distracted but overall polite.\n"
                        "- I felt the rider was not very engaging, but the delivery was smooth.\n"
                        "\n\n"
                        "3. Moderate: The comment indicates noticeable issues with the delivery partner's behavior, affecting the customer's satisfaction.\n"
                        "Examples:\n"
                        "- The rider seemed impatient and was not very courteous.\n"
                        "- There was a lack of communication from the rider when I tried to ask questions.\n"
                        "- The rider disregarded my instructions regarding a safe spot for delivery.\n"
                        "\n\n"
                        "4. Significant: The comment reports serious issues that suggest the delivery partner's behavior was inappropriate or unprofessional and significantly impacted the customer experience.\n"
                        "Examples:\n"
                        "- The rider made an inappropriate comment during the delivery.\n"
                        "- The rider's behavior was unprofessional and made me uncomfortable.\n"
                        "- I felt uneasy as the rider ignored my request to leave the package at the door.\n"
                        "\n\n"
                        "5. Severe: The comment describes very alarming behavior, potentially involving unsafe, unethical or extremely unprofessional actions by the delivery partner.\n"
                        "Examples:\n"
                        "- The rider harassed me, making the experience very distressing.\n"
                        "- I was verbally assaulted by the rider, which was frightening.\n"
                        "- The rider behaved aggressively, and I felt threatened.\n"
                        "\n\n"
                    )
                },
                {"role": "user", "content": feedback},
            ],
            temperature=0,
        )
        # pdb.set_trace()
        content = response.choices[0].message.content.strip()
        score = extract_score(content)
        if score is None:
            # pdb.set_trace()
            logger.info("This is an info log when score is none")
            print(f"⚠️ Couldn't extract score from: {content}")
        return score
    except Exception as e:
        # pdb.set_trace()
        logger.info("This is an info log when exception has occurred")
        print(f"❌ Error for: {feedback[:30]}... → {e}")
        return None


async def process_all_feedback(feedbacks):
    tasks = [get_score_async(fb) for fb in feedbacks]
    return await asyncio.gather(*tasks)


def classify_feedbacks():
    # pdb.set_trace()
    logger.info("This is an info log inside classify feedbacks")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            logger.info("reading csv")
            if "feedback" not in df.columns:
                st.error("CSV must contain a 'feedback' column.")
            else:
                # Clean feedback
                df["feedback"] = df["feedback"].astype(str)
                df = df[df["feedback"].str.strip().str.len() > 3]
                logger.info("***** cleaning feedback *******")

                if st.button("Score Feedback"):
                    with st.spinner("Scoring feedbacks... please wait ⏳"):
                        scores = asyncio.run(process_all_feedback(df["feedback"].tolist()))
                        df["misconduct_score"] = scores
                        st.success("✅ Done! See below:")
                        st.dataframe(df)

                        buffer = BytesIO()
                        df.to_csv(buffer, index=False)
                        buffer.seek(0)

                        st.download_button(
                            label="📥 Download CSV with Scores",
                            data=buffer,
                            file_name="scored_feedbacks.csv",
                            mime="text/csv",
                        )

        except Exception as e:
            logger.info("This is an info log when exception has occurred before reading csv")
            st.error(f"Something went wrong: {e}")
    else:
        logger.info("This is an info log when uploaded file is not there")