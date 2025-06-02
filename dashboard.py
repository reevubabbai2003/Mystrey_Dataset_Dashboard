import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import re

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.dowload('punkt_tabs')

# Set default color scheme for all plots
px.defaults.color_discrete_sequence = px.colors.qualitative.Plotly
px.defaults.template = "plotly_white"

# Title of the dashboard
st.title("Applicant Data Dashboard")

# Load dataset with proper skill handling
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_dataset.csv")
    
    # Convert to string and handle NaN values
    df['Skill_1'] = df['Skill_1'].fillna('').astype(str)
    df['Skill_2'] = df['Skill_2'].fillna('').astype(str)
    df['Experience'] = df['Experience'].astype(str)
    
    # Clean skill values
    df['Skill_1'] = df['Skill_1'].apply(lambda x: x.strip())
    df['Skill_2'] = df['Skill_2'].apply(lambda x: x.strip())
    
    return df

df = load_data()

# Function to extract keywords
def extract_keywords(text_series, top_n=15):
    # Filter out empty strings
    non_empty_text = text_series[text_series != '']
    if non_empty_text.empty:
        return []
        
    combined_text = ' '.join(non_empty_text.astype(str))
    tokens = word_tokenize(combined_text.lower())
    stop_words = set(stopwords.words('english'))
    
    # Add custom stopwords
    custom_stopwords = {'role', 'position', 'job', 'applicant', 'application', 
                        'applied', 'apply', 'applicants', 'seeking', 'desired',
                        'wanted', 'looking', 'for', 'position', 'positions'}
    stop_words.update(custom_stopwords)
    
    # Filter out stopwords and invalid tokens
    keywords = [
        token for token in tokens 
        if token not in stop_words 
        and re.match(r'^[a-zA-Z]{3,}$', token)
    ]
    
    keyword_counts = Counter(keywords)
    return keyword_counts.most_common(top_n)

# Sidebar filters
st.sidebar.header("Filter Applicants")

# Gender filter
genders = df['Gender'].unique().tolist()
selected_genders = st.sidebar.multiselect(
    "Gender",
    options=genders,
    default=genders
)

# Country filter
country_counts_full = df['Country'].value_counts()
sorted_countries = country_counts_full.index.tolist() 
top_10_countries = sorted_countries[:10]
selected_countries = st.sidebar.multiselect(
    "Country (top-down)",
    options=sorted_countries,
    default=top_10_countries
)

# Employment Type filter
employment_types = df['Employment_type'].unique().tolist()
selected_employment = st.sidebar.multiselect(
    "Employment Type",
    options=employment_types,
    default=employment_types
)

# Experience filter
experience_options = ['Yes', 'No']
selected_experience = st.sidebar.multiselect(
    "Experience",
    options=experience_options,
    default=experience_options
)

# Separate Skill Filters
st.sidebar.subheader("Skill Filters")

# Skill 1 filter
unique_skill1 = sorted(df['Skill_1'].unique().tolist())
selected_skill1 = st.sidebar.multiselect(
    "Skill 1",
    options=unique_skill1
)

# Skill 2 filter
unique_skill2 = sorted(df['Skill_2'].unique().tolist())
selected_skill2 = st.sidebar.multiselect(
    "Skill 2",
    options=unique_skill2
)

# Apply filters to dataframe
df_filtered = df[
    (df['Gender'].isin(selected_genders)) &
    (df['Country'].isin(selected_countries)) &
    (df['Employment_type'].isin(selected_employment)) &
    (df['Experience'].isin(selected_experience))
]

# Apply skill filters if any are selected
if selected_skill1:
    df_filtered = df_filtered[df_filtered['Skill_1'].isin(selected_skill1)]
    
if selected_skill2:
    df_filtered = df_filtered[df_filtered['Skill_2'].isin(selected_skill2)]

# --- Visualizations ---

st.header("General Distributions")

# Create columns for side-by-side plots
col1, col2 = st.columns(2)

with col1:
    # Age distribution with color
    fig_age = px.histogram(
        df_filtered,
        x="Age",
        nbins=10,
        title="Age Distribution",
        color_discrete_sequence=['#1f77b4']
    )
    st.plotly_chart(fig_age)

with col2:
    # Monthly family income distribution with color
    fig_income = px.histogram(
        df_filtered,
        x="Monthly_Family_Income",
        nbins=10,
        title="Monthly Family Income Distribution",
        color_discrete_sequence=['#ff7f0e']
    )
    st.plotly_chart(fig_income)

st.header("Category Distributions")

# Create columns for pie charts
col1, col2 = st.columns(2)

with col1:
    # Gender distribution pie chart
    gender_counts = df_filtered['Gender'].value_counts().reset_index()
    fig_gender = px.pie(
        gender_counts,
        names='Gender',
        values='count',
        title="Gender Distribution",
        color='Gender'
    )
    st.plotly_chart(fig_gender)

with col2:
    # Marital status distribution pie chart
    marital_counts = df_filtered['Marital_Status'].value_counts().reset_index()
    fig_marital = px.pie(
        marital_counts,
        names='Marital_Status',
        values='count',
        title="Marital Status Distribution",
        color='Marital_Status'
    )
    st.plotly_chart(fig_marital)

# Create another set of columns for pie charts
col3, col4 = st.columns(2)

with col3:
    # Employment type distribution pie chart
    employment_counts = df_filtered['Employment_type'].value_counts().reset_index()
    fig_employment = px.pie(
        employment_counts,
        names='Employment_type',
        values='count',
        title="Employment Type Distribution",
        color='Employment_type'
    )
    st.plotly_chart(fig_employment)

with col4:
    # Experience distribution pie chart
    experience_counts = df_filtered['Experience'].value_counts().reset_index()
    fig_experience = px.pie(
        experience_counts,
        names='Experience',
        values='count',
        title="Experience Distribution",
        color='Experience'
    )
    st.plotly_chart(fig_experience)

st.header("Role and Skills Frequency")

# Top Roles bar chart with color
role_counts = df_filtered['Applied_role'].value_counts().nlargest(10).reset_index()
fig_role = px.bar(
    role_counts,
    x='Applied_role',
    y='count',
    title="Top Roles Applied For",
    color='Applied_role'
)
st.plotly_chart(fig_role)

# Keyword Analysis of Applied Roles with color
st.subheader("Keyword Analysis of Applied Roles")

# Extract keywords from applied roles
keyword_counts = extract_keywords(df_filtered['Applied_role'])

if keyword_counts:
    keywords_df = pd.DataFrame(keyword_counts, columns=['Keyword', 'Count'])
    fig_keywords = px.bar(
        keywords_df,
        x='Keyword',
        y='Count',
        title="Top Keywords in Applied Roles",
        color='Count',
        color_continuous_scale='Bluered'
    )
    st.plotly_chart(fig_keywords)
else:
    st.warning("No keywords found in applied roles after filtering")

# Side-by-side Skill distributions
st.subheader("Skill Distribution Comparison")
skill_col1, skill_col2 = st.columns(2)

with skill_col1:
    # Skill 1 distribution
    skill1_counts = df_filtered['Skill_1'].value_counts().reset_index()
    fig_skill1 = px.bar(
        skill1_counts,
        x='Skill_1',
        y='count',
        title="Skill 1 Distribution",
        color='Skill_1'
    )
    st.plotly_chart(fig_skill1)

with skill_col2:
    # Skill 2 distribution
    skill2_counts = df_filtered['Skill_2'].value_counts().reset_index()
    fig_skill2 = px.bar(
        skill2_counts,
        x='Skill_2',
        y='count',
        title="Skill 2 Distribution",
        color='Skill_2'
    )
    st.plotly_chart(fig_skill2)

st.header("Location Distribution")

# Country-wise distribution with color
country_counts = df_filtered['Country'].value_counts().reset_index()
fig_country = px.bar(
    country_counts,
    x='Country',
    y='count',
    title="Applicants by Country (Filtered)",
    color='Country'
)
st.plotly_chart(fig_country)

st.header("Filtered Data Table")
st.dataframe(df_filtered)   
