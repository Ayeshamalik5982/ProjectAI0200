import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# --------------------------------------------------
# PAGE CONFIG & THEME (Neutral, Easy on Eyes)
# --------------------------------------------------
st.set_page_config(
    page_title="Netflix ML Dashboard",
    layout="wide"
)

st.markdown("""
<style>
body {
    background-color: #f2f3f7;
    color: #333333;
}
.stApp {
    background-color: #f2f3f7;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA WITH SPECIFIED ENCODING
# --------------------------------------------------
@st.cache_data
def load_data():
    # Specify encoding to avoid UnicodeDecodeError
    return pd.read_csv("netflix_titles.csv", encoding='ISO-8859-1')

df = load_data()

st.title("🎬 Netflix Data Analysis & ML App")

# --------------------------------------------------
# RAW DATA PREVIEW
# --------------------------------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------
df = df.dropna(axis=1, how='all')

df['duration_num'] = df['duration'].str.extract('(\d+)').astype(float)

df_model = df.drop(columns=[
    'show_id', 'title', 'director', 'cast',
    'country', 'date_added', 'description'
])

# Encode target
le = LabelEncoder()
df_model['type'] = le.fit_transform(df_model['type'])

# Encode categorical features
for col in ['rating', 'listed_in']:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

# Handle missing values
numeric_cols = df_model.select_dtypes(include=['int64', 'float64']).columns
df_model[numeric_cols] = df_model[numeric_cols].fillna(
    df_model[numeric_cols].median()
)

# --------------------------------------------------
# TARGET COLUMN VISUALIZATION
# --------------------------------------------------
st.subheader("🎯 Target Variable Distribution (Movie vs TV Show)")

fig, ax = plt.subplots()
sns.countplot(x=df['type'], ax=ax)
ax.set_xlabel("Type")
ax.set_ylabel("Count")
st.pyplot(fig)

# --------------------------------------------------
# BAR CHART
# --------------------------------------------------
st.subheader("📊 Bar Chart: Rating Distribution")

fig, ax = plt.subplots()
df['rating'].value_counts().plot(kind='bar', ax=ax, color="#4c72b0")
ax.set_xlabel("Rating")
ax.set_ylabel("Count")
st.pyplot(fig)

# --------------------------------------------------
# HISTOGRAM
# --------------------------------------------------
st.subheader("📈 Histogram: Duration Distribution")

fig, ax = plt.subplots()
ax.hist(df_model['duration_num'], bins=30, color="#55a868")
ax.set_xlabel("Duration")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# --------------------------------------------------
# SCATTER PLOT
# --------------------------------------------------
st.subheader("🔵 Scatter Plot: Release Year vs Duration")

fig, ax = plt.subplots()
ax.scatter(df_model['release_year'], df_model['duration_num'], alpha=0.5)
ax.set_xlabel("Release Year")
ax.set_ylabel("Duration")
st.pyplot(fig)

# --------------------------------------------------
# HEATMAP
# --------------------------------------------------
st.subheader("🔥 Correlation Heatmap")

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    df_model.corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax
)
st.pyplot(fig)

# --------------------------------------------------
# MACHINE LEARNING MODEL
# --------------------------------------------------
st.subheader("🤖 Machine Learning Model (Logistic Regression)")

X = df_model.drop('type', axis=1)
y = df_model['type']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# --------------------------------------------------
# SIDEBAR PREDICTION
# --------------------------------------------------
st.sidebar.header("🎯 Predict Content Type")

duration = st.sidebar.number_input("Duration (minutes)", min_value=1, value=90)
release_year = st.sidebar.number_input("Release Year", min_value=1940, value=2020)
rating = st.sidebar.selectbox("Rating", sorted(df['rating'].unique()))

rating_encoded = le.fit_transform(df[['rating']].astype(str))[:1]

input_data = np.array([[release_year, rating_encoded[0], duration]])

prediction = model.predict(scaler.transform(input_data))

result = "TV Show" if prediction[0] == 1 else "Movie"
st.sidebar.success(f"Predicted Type: {result}")

