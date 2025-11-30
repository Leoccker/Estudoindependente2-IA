import pandas as pd
import re


def clean_text(text):
    if pd.isna(text):
        return ""

    text = text.lower()
    text = text.replace("subject: ", "")
    text = text.replace("re : ", "")
    text = text.replace("fw : ", "")
    text = text.replace("fwd : ", "")
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\d+", " number ", text)
    text = re.sub(r"[^a-z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


df1 = pd.read_csv("spam0.csv", encoding="latin-1")
df2 = pd.read_csv("spam1.csv", encoding="latin-1")
df3 = pd.read_csv("spam2.csv", encoding="latin-1")
df4 = pd.read_csv("spam3.csv", encoding="latin-1")
df5 = pd.read_csv("spam4.csv", encoding="latin-1")

df1 = df1[["Category", "Message"]]
df2 = df2[["label", "text"]]
df3 = df3[["label", "email"]]
df4 = df4[["spam", "text"]]
df5 = df5[["CLASS", "CONTENT"]]

df = pd.concat(
    [
        df1.rename(columns={"Category": "Category", "Message": "Message"}),
        df2.rename(columns={"label": "Category", "text": "Message"}),
        df3.rename(columns={"label": "Category", "email": "Message"}),
        df4.rename(columns={"spam": "Category", "text": "Message"}),
        df5.rename(columns={"CLASS": "Category", "CONTENT": "Message"}),
    ],
    ignore_index=True,
)

df["Category"] = df["Category"].replace(
    {
        "ham": "ham",
        "spam": "spam",
        "0": "ham",
        "1": "spam",
        0: "ham",
        1: "spam",
    }
)

df['Message'] = df['Message'].apply(clean_text)

df = df.dropna(subset=['Message'])
df = df[df['Message'].str.strip() != ""]

df.to_csv("spam.csv", index=False)
