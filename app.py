import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score, accuracy_score
)

# Page configuration
st.set_page_config(
    page_title="Customer Churn Analysis & Prediction | Oscar Ochoa",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark purple gradient theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f0f23 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d1b4e 0%, #1a1a2e 100%);
        border-right: 1px solid #6c4ab6;
    }

    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(90deg, #b794f6, #8b5cf6, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Metric containers */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #2d1b4e 0%, #1e1e3f 100%);
        border: 1px solid #6c4ab6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.2);
    }

    /* Cards/Containers */
    .stExpander {
        background: linear-gradient(135deg, #1e1e3f 0%, #2d1b4e 100%);
        border: 1px solid #6c4ab6;
        border-radius: 10px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #8b5cf6, #6c4ab6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #a78bfa, #8b5cf6);
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4);
        transform: translateY(-2px);
    }

    /* Select boxes and inputs */
    .stSelectbox > div > div, .stMultiSelect > div > div {
        background-color: #1e1e3f;
        border: 1px solid #6c4ab6;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #2d1b4e 0%, #1e1e3f 100%);
        border: 1px solid #6c4ab6;
        border-radius: 8px 8px 0 0;
        color: #a78bfa;
        padding: 10px 20px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8b5cf6 0%, #6c4ab6 100%);
        color: white;
    }

    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #2d1b4e 0%, #1e1e3f 100%);
        border-left: 4px solid #8b5cf6;
        padding: 20px;
        border-radius: 0 10px 10px 0;
        margin: 10px 0;
    }

    /* Highlight box */
    .highlight-box {
        background: linear-gradient(135deg, #4c1d95 0%, #2d1b4e 100%);
        border: 1px solid #8b5cf6;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.3);
    }

    /* Data editor and dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid #6c4ab6;
        border-radius: 10px;
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #8b5cf6, #a78bfa);
    }

    /* Divider */
    hr {
        border-color: #6c4ab6;
    }

    /* Custom metric style */
    .custom-metric {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #2d1b4e 0%, #1e1e3f 100%);
        border: 1px solid #6c4ab6;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.2);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #b794f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        color: #a78bfa;
        font-size: 1rem;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Color scheme for charts
COLORS = {
    'primary': '#8b5cf6',
    'secondary': '#a78bfa',
    'accent': '#c4b5fd',
    'success': '#10b981',
    'danger': '#ef4444',
    'warning': '#f59e0b',
    'background': '#1a1a2e',
    'surface': '#2d1b4e',
    'text': '#e2e8f0'
}

PLOTLY_TEMPLATE = {
    'layout': {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'color': COLORS['text']},
        'xaxis': {'gridcolor': 'rgba(139, 92, 246, 0.2)', 'zerolinecolor': 'rgba(139, 92, 246, 0.3)'},
        'yaxis': {'gridcolor': 'rgba(139, 92, 246, 0.2)', 'zerolinecolor': 'rgba(139, 92, 246, 0.3)'}
    }
}


@st.cache_data
def load_data():
    """Load and preprocess the dataset."""
    df = pd.read_excel("CST-570-RS-WAFn-UseC-Telco-Customer-Churn.xlsx")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df


@st.cache_resource
def train_models(df):
    """Train all models and return them with metrics."""
    # Prepare data
    feature_cols = [
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges"
    ]

    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_cols = [col for col in feature_cols if col not in num_cols]

    X = df[feature_cols].copy()
    y = df['Churn'].map({'Yes': 1, 'No': 0}).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Preprocessors
    preprocessor = ColumnTransformer([
        ("num", 'passthrough', num_cols),
        ('cat', OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols)
    ])

    preprocessor_scaled = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop='first'), cat_cols)
    ])

    # Models
    models = {
        'Logistic Regression': Pipeline([
            ('process', preprocessor),
            ('model', LogisticRegression(max_iter=3000))
        ]),
        'K-Nearest Neighbors': Pipeline([
            ('process', preprocessor_scaled),
            ('model', KNeighborsClassifier())
        ]),
        'Decision Tree': Pipeline([
            ('process', ColumnTransformer([
                ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols)
            ])),
            ('model', DecisionTreeClassifier(max_depth=7))
        ]),
        'Random Forest': Pipeline([
            ('process', preprocessor),
            ('model', RandomForestClassifier(n_estimators=100, max_depth=7))
        ])
    }

    results = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)

        results[name] = {
            'pipeline': pipeline,
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'fpr': fpr,
            'tpr': tpr,
            'y_test': y_test,
            'y_pred': y_pred
        }

    return results, feature_cols, num_cols, cat_cols


def create_gauge_chart(value, title, max_val=100):
    """Create a gauge chart for metrics."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'color': COLORS['text']}},
        number={'suffix': '%', 'font': {'color': COLORS['primary']}},
        gauge={
            'axis': {'range': [0, max_val], 'tickcolor': COLORS['text']},
            'bar': {'color': COLORS['primary']},
            'bgcolor': COLORS['surface'],
            'bordercolor': COLORS['primary'],
            'steps': [
                {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.3)'},
                {'range': [50, 75], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [75, 100], 'color': 'rgba(16, 185, 129, 0.3)'}
            ],
            'threshold': {
                'line': {'color': COLORS['accent'], 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=250
    )
    return fig


def main():
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1 style='font-size: 1.5rem;'>Churn Analysis</h1>
            <p style='color: #a78bfa;'>ML-Powered Predictions</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["Overview", "Explore the Data", "What I Found", "How the Models Did", "Try It Yourself", "Takeaways"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # About the Author section
        st.markdown("""
        <div style='padding: 15px; background: linear-gradient(135deg, #1e1e3f 0%, #2d1b4e 100%); border-radius: 10px; border: 1px solid #6c4ab6;'>
            <h4 style='color: #a78bfa; margin-bottom: 10px; text-align: center;'>About Me</h4>
            <p style='color: #e2e8f0; font-size: 0.85rem; line-height: 1.5;'>
                Hey, I'm <strong>Oscar Ochoa</strong>, a software enthusiast with a passion for turning raw data into meaningful stories.
                What draws me to data science is the thrill of diving into a new dataset and uncovering insights that drive real business decisions.
            </p>
            <br>
            <p style='color: #a78bfa; font-size: 0.8rem;'>
                <strong>M.S. Computer Science</strong><br>
                <span style='color: #9ca3af;'>Grand Canyon University • July 2026</span>
            </p>
            <p style='color: #a78bfa; font-size: 0.8rem; margin-top: 8px;'>
                <strong>B.S. Computer Science</strong><br>
                <span style='color: #9ca3af;'>Cal State Monterey Bay • 2024</span>
            </p>
            <br>
            <p style='color: #a78bfa; font-size: 0.8rem;'>
                <strong>Seeking Opportunities In:</strong><br>
                <span style='color: #9ca3af;'>Software Engineering • Data Science<br>Machine Learning • Data Analytics</span>
            </p>
            <br>
            <div style='text-align: center; margin-top: 10px;'>
                <a href='https://github.com/o-o-dev' target='_blank' style='color: #a78bfa; text-decoration: none; margin-right: 15px; font-size: 1.2rem;'>
                    <img src='https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg' width='24' style='filter: invert(70%) sepia(50%) saturate(500%) hue-rotate(220deg);'/>
                </a>
                <a href='https://www.linkedin.com/in/oscar-ochoa-096420224/' target='_blank' style='color: #a78bfa; text-decoration: none; font-size: 1.2rem;'>
                    <img src='https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg' width='24'/>
                </a>
            </div>
            <br>
            <p style='color: #9ca3af; font-size: 0.75rem; font-style: italic; text-align: center; border-top: 1px solid #6c4ab6; padding-top: 10px; margin-top: 5px;'>
                "Data Science is not a philosophy, there is never one correct answer for everything."
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div style='text-align: center; padding: 10px;'>
            <p style='color: #6c4ab6; font-size: 0.75rem;'>Built with Streamlit</p>
            <p style='color: #6c4ab6; font-size: 0.75rem;'>Python • Scikit-learn • Plotly</p>
        </div>
        """, unsafe_allow_html=True)

    # Load data
    df = load_data()
    results, feature_cols, num_cols, cat_cols = train_models(df)

    # Page routing
    if page == "Overview":
        show_overview(df)
    elif page == "Explore the Data":
        show_data_explorer(df)
    elif page == "What I Found":
        show_eda(df)
    elif page == "How the Models Did":
        show_models(results)
    elif page == "Try It Yourself":
        show_predictor(results, feature_cols, num_cols, cat_cols, df)
    elif page == "Takeaways":
        show_findings(results)


def show_overview(df):
    """Display the overview dashboard."""
    st.markdown("""
    <h1 style='text-align: center; font-size: 2.8rem; margin-bottom: 0;'>Customer Churn Analysis & Prediction</h1>
    <p style='text-align: center; color: #a78bfa; font-size: 1.2rem; margin-top: 10px;'>
        Predicting Customer Attrition with Machine Learning
    </p>
    <p style='text-align: center; color: #6c4ab6; font-size: 1rem; margin-top: 5px;'>
        By Oscar Ochoa
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Key metrics
    total_customers = len(df)
    churn_rate = (df['Churn'] == 'Yes').mean() * 100
    avg_tenure = df['tenure'].mean()
    avg_monthly = df['MonthlyCharges'].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class='custom-metric'>
            <div class='metric-value'>{total_customers:,}</div>
            <div class='metric-label'>Total Customers</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='custom-metric'>
            <div class='metric-value'>{churn_rate:.1f}%</div>
            <div class='metric-label'>Churn Rate</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='custom-metric'>
            <div class='metric-value'>{avg_tenure:.0f}</div>
            <div class='metric-label'>Avg Tenure (months)</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class='custom-metric'>
            <div class='metric-value'>${avg_monthly:.0f}</div>
            <div class='metric-label'>Avg Monthly Charges</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        # Churn distribution
        churn_counts = df['Churn'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=['Retained', 'Churned'],
            values=[churn_counts.get('No', 0), churn_counts.get('Yes', 0)],
            hole=0.6,
            marker=dict(colors=[COLORS['success'], COLORS['danger']]),
            textinfo='percent+label',
            textfont=dict(color='white')
        )])
        fig.update_layout(
            title={'text': 'Customer Retention', 'font': {'color': COLORS['text']}},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': COLORS['text']},
            showlegend=False,
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Contract distribution
        contract_counts = df['Contract'].value_counts()
        fig = go.Figure(data=[go.Bar(
            x=contract_counts.index,
            y=contract_counts.values,
            marker=dict(
                color=[COLORS['danger'], COLORS['warning'], COLORS['success']],
                line=dict(color=COLORS['primary'], width=2)
            )
        )])
        fig.update_layout(
            title={'text': 'Contract Distribution', 'font': {'color': COLORS['text']}},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': COLORS['text']},
            xaxis={'title': 'Contract Type'},
            yaxis={'title': 'Count'},
            height=350
        )
        fig.update_xaxes(gridcolor='rgba(139, 92, 246, 0.2)')
        fig.update_yaxes(gridcolor='rgba(139, 92, 246, 0.2)')
        st.plotly_chart(fig, use_container_width=True)

    # Why I Built This
    st.markdown("""
    <div class='highlight-box'>
        <h3>Why I Built This</h3>
        <p>I recently discovered Streamlit and I wanted to have a go at it. Data can tell a million stories,
        you just have to know where to look and have the right tools to help you uncover interesting qualities.
        I built this to show that businesses can extract valuable information just by getting your hands a little dirty.
        These kinds of insights can help your business make informed decisions and understand customers and their needs.
        I hope you have fun looking through some of the interesting stories this dataset has.</p>
        <br>
        <h4>Models I Used:</h4>
        <ul>
            <li><strong>Logistic Regression</strong> - Interpretable baseline model</li>
            <li><strong>K-Nearest Neighbors</strong> - Distance-based classification</li>
            <li><strong>Decision Tree</strong> - Rule-based interpretable model</li>
            <li><strong>Random Forest</strong> - Ensemble method for robust predictions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def show_data_explorer(df):
    """Display interactive data explorer."""
    st.markdown("<h1>Explore the Data</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        <p>Explore the Telco customer dataset with <strong>7,043 customers</strong> and <strong>21 features</strong>.
        Filter and analyze the data interactively.</p>
    </div>
    """, unsafe_allow_html=True)

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        churn_filter = st.multiselect(
            "Churn Status",
            options=df['Churn'].unique(),
            default=df['Churn'].unique()
        )

    with col2:
        contract_filter = st.multiselect(
            "Contract Type",
            options=df['Contract'].unique(),
            default=df['Contract'].unique()
        )

    with col3:
        internet_filter = st.multiselect(
            "Internet Service",
            options=df['InternetService'].unique(),
            default=df['InternetService'].unique()
        )

    # Apply filters
    filtered_df = df[
        (df['Churn'].isin(churn_filter)) &
        (df['Contract'].isin(contract_filter)) &
        (df['InternetService'].isin(internet_filter))
    ]

    st.markdown(f"**Showing {len(filtered_df):,} of {len(df):,} customers**")

    # Data table
    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=400
    )

    # Column statistics
    st.markdown("### Column Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Numeric Columns**")
        st.dataframe(filtered_df[['tenure', 'MonthlyCharges', 'TotalCharges']].describe())

    with col2:
        st.markdown("**Data Types**")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null': df.count().values
        })
        st.dataframe(dtype_df, height=300)


def show_eda(df):
    """Display EDA visualizations."""
    st.markdown("<h1>What I Found</h1>", unsafe_allow_html=True)

    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Churn Factors", "Financial Analysis", "Services"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Tenure distribution
            fig = px.histogram(
                df, x='tenure', color='Churn',
                nbins=30, barmode='overlay',
                color_discrete_map={'Yes': COLORS['danger'], 'No': COLORS['success']},
                title='Tenure Distribution by Churn Status'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': COLORS['text']}
            )
            fig.update_xaxes(gridcolor='rgba(139, 92, 246, 0.2)')
            fig.update_yaxes(gridcolor='rgba(139, 92, 246, 0.2)')
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div class='info-box'>
                <strong>Key Insight:</strong> Churn is heavily concentrated at low tenure. Customers who have been
                with the company for a short time tend to churn way more than long-term customers.<br><br>
                <strong>What this means for modeling:</strong> Tenure is a strong predictor and will likely have a
                <em>negative coefficient</em> in logistic regression. As tenure goes up, churn probability goes down.
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Monthly charges distribution
            fig = px.box(
                df, x='Churn', y='MonthlyCharges',
                color='Churn',
                color_discrete_map={'Yes': COLORS['danger'], 'No': COLORS['success']},
                title='Monthly Charges by Churn Status'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': COLORS['text']}
            )
            fig.update_xaxes(gridcolor='rgba(139, 92, 246, 0.2)')
            fig.update_yaxes(gridcolor='rgba(139, 92, 246, 0.2)')
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div class='info-box'>
                <strong>Key Insight:</strong> Churned customers pay more monthly on average. Non-churners mostly
                pay $20-$86/month, while churners concentrate between $57-$90/month.<br><br>
                <strong>What this means for modeling:</strong> Monthly charges will likely have a <em>positive coefficient</em>.
                As charges go up, so does churn probability. Lower-paying customers probably only use essential services,
                so they're less likely to leave.
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            # Contract type vs Churn
            contract_churn = df.groupby(['Contract', 'Churn']).size().unstack(fill_value=0)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Retained',
                x=contract_churn.index,
                y=contract_churn['No'],
                marker_color=COLORS['success']
            ))
            fig.add_trace(go.Bar(
                name='Churned',
                x=contract_churn.index,
                y=contract_churn['Yes'],
                marker_color=COLORS['danger']
            ))
            fig.update_layout(
                barmode='group',
                title='Churn by Contract Type',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': COLORS['text']}
            )
            fig.update_xaxes(gridcolor='rgba(139, 92, 246, 0.2)')
            fig.update_yaxes(gridcolor='rgba(139, 92, 246, 0.2)')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Payment method vs Churn
            payment_churn = df.groupby(['PaymentMethod', 'Churn']).size().unstack(fill_value=0)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Retained',
                x=payment_churn.index,
                y=payment_churn['No'],
                marker_color=COLORS['success']
            ))
            fig.add_trace(go.Bar(
                name='Churned',
                x=payment_churn.index,
                y=payment_churn['Yes'],
                marker_color=COLORS['danger']
            ))
            fig.update_layout(
                barmode='group',
                title='Churn by Payment Method',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': COLORS['text']},
                xaxis_tickangle=-45
            )
            fig.update_xaxes(gridcolor='rgba(139, 92, 246, 0.2)')
            fig.update_yaxes(gridcolor='rgba(139, 92, 246, 0.2)')
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class='highlight-box'>
            <h4>Key Churn Factors Identified:</h4>
            <ul>
                <li><strong>Month-to-month contracts</strong> have the highest churn rate. Interestingly, even within
                month-to-month, more customers stay than leave. But the <em>proportion</em> who churn is much higher
                than yearly contracts. Getting customers to commit to at least one year dramatically reduces churn.</li>
                <li><strong>Electronic check</strong> payment method shows elevated churn. This is a behavioral signal.
                Customers paying via e-check may be experiencing payment friction or are less committed. Encouraging
                automatic payment methods could help with retention.</li>
                <li><strong>Fiber optic</strong> internet customers churn more than DSL users. This could point to
                dissatisfaction with service quality or pricing. Fiber is typically more expensive, which compounds
                with the monthly charges effect.</li>
                <li><strong>Short tenure</strong> customers are most at risk. The critical retention window is the
                first 12 months.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        # Scatter plot: Tenure vs Monthly Charges
        fig = px.scatter(
            df, x='tenure', y='MonthlyCharges', color='Churn',
            color_discrete_map={'Yes': COLORS['danger'], 'No': COLORS['success']},
            title='Monthly Charges vs Tenure (Colored by Churn)',
            opacity=0.6
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': COLORS['text']},
            height=500
        )
        fig.update_xaxes(gridcolor='rgba(139, 92, 246, 0.2)')
        fig.update_yaxes(gridcolor='rgba(139, 92, 246, 0.2)')
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class='info-box'>
                <strong>Key Insight:</strong> The majority of churned customers are concentrated in the
                <strong>0-10 month tenure range with high monthly payments</strong>. This is the highest risk segment.<br><br>
                <strong>Total Charges Pattern:</strong> Most churned customers have very low total charges.
                They leave before accumulating significant payments. This confirms that customers who pay more
                monthly tend to leave sooner. Total charges will have a <em>negative coefficient</em> in our model.
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class='info-box'>
                <strong>The Retention Paradox:</strong> Customers who experience the service longer tend to
                stay, but we need to get them past that critical first year. The question becomes:
                <em>how do we get new, high-paying customers to commit before they churn?</em><br><br>
                <strong>Recommendation:</strong> Target high-value new customers with retention incentives
                within their first 6 months.
            </div>
            """, unsafe_allow_html=True)

    with tab4:
        # Internet service analysis
        col1, col2 = st.columns(2)

        with col1:
            internet_churn = df.groupby(['InternetService', 'Churn']).size().unstack(fill_value=0)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Retained',
                x=internet_churn.index,
                y=internet_churn['No'],
                marker_color=COLORS['success']
            ))
            fig.add_trace(go.Bar(
                name='Churned',
                x=internet_churn.index,
                y=internet_churn['Yes'],
                marker_color=COLORS['danger']
            ))
            fig.update_layout(
                barmode='group',
                title='Churn by Internet Service Type',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': COLORS['text']}
            )
            fig.update_xaxes(gridcolor='rgba(139, 92, 246, 0.2)')
            fig.update_yaxes(gridcolor='rgba(139, 92, 246, 0.2)')
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div class='info-box'>
                <strong>Fiber Optic Problem:</strong> Despite being the most popular service, fiber optic has
                disproportionately high churn. This could indicate service quality issues, pricing concerns,
                or unmet customer expectations. DSL and no-internet customers show much lower churn rates.
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Services count vs churn
            services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies']
            df_temp = df.copy()
            df_temp['num_services'] = (df_temp[services] == 'Yes').sum(axis=1)

            fig = px.box(
                df_temp, x='Churn', y='num_services', color='Churn',
                color_discrete_map={'Yes': COLORS['danger'], 'No': COLORS['success']},
                title='Number of Services vs Churn'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': COLORS['text']}
            )
            fig.update_xaxes(gridcolor='rgba(139, 92, 246, 0.2)')
            fig.update_yaxes(gridcolor='rgba(139, 92, 246, 0.2)')
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div class='info-box'>
                <strong>Service Bundling Effect:</strong> Customers with more services tend to stay longer.
                This makes sense because finding a new provider that offers all their services is harder.
                <strong>Recommendation:</strong> Encourage customers to add services early in their tenure to increase stickiness.
            </div>
            """, unsafe_allow_html=True)


def show_models(results):
    """Display model performance comparison."""
    st.markdown("<h1>How the Models Did</h1>", unsafe_allow_html=True)

    # Class imbalance warning
    st.markdown("""
    <div class='highlight-box'>
        <h4>Important: Class Imbalance Challenge</h4>
        <p>Our dataset has <strong>73.5% retained customers</strong> vs <strong>26.5% churned customers</strong>.
        This imbalance affects all models. They tend to be better at predicting the majority class (no churn)
        and struggle with the minority class (churn). That's why we use <strong>ROC AUC</strong> alongside
        accuracy to evaluate performance.</p>
    </div>
    """, unsafe_allow_html=True)

    # Model selector
    selected_model = st.selectbox(
        "Select a model to view details:",
        list(results.keys())
    )

    # Overview metrics
    st.markdown("### Performance Overview")

    cols = st.columns(4)
    for i, (name, data) in enumerate(results.items()):
        with cols[i]:
            is_selected = name == selected_model
            border_color = COLORS['primary'] if is_selected else '#4a4a6a'
            st.markdown(f"""
            <div style='
                text-align: center;
                padding: 15px;
                background: linear-gradient(135deg, #2d1b4e 0%, #1e1e3f 100%);
                border: 2px solid {border_color};
                border-radius: 10px;
                {"box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4);" if is_selected else ""}
            '>
                <div style='font-weight: bold; color: #a78bfa;'>{name}</div>
                <div style='font-size: 1.5rem; color: white; margin-top: 5px;'>{data['accuracy']*100:.1f}%</div>
                <div style='color: #6c4ab6; font-size: 0.8rem;'>Accuracy</div>
                <div style='font-size: 1.2rem; color: #8b5cf6; margin-top: 5px;'>{data['roc_auc']:.3f}</div>
                <div style='color: #6c4ab6; font-size: 0.8rem;'>ROC AUC</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Detailed view of selected model
    col1, col2 = st.columns(2)

    with col1:
        # Gauge chart for accuracy
        fig = create_gauge_chart(
            results[selected_model]['accuracy'] * 100,
            f"{selected_model} Accuracy"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # ROC Curve
        fig = go.Figure()

        # Add ROC curve for selected model
        fig.add_trace(go.Scatter(
            x=results[selected_model]['fpr'],
            y=results[selected_model]['tpr'],
            name=f'{selected_model} (AUC={results[selected_model]["roc_auc"]:.3f})',
            line=dict(color=COLORS['primary'], width=3)
        ))

        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random Classifier',
            line=dict(color='gray', dash='dash')
        ))

        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': COLORS['text']},
            height=300
        )
        fig.update_xaxes(gridcolor='rgba(139, 92, 246, 0.2)')
        fig.update_yaxes(gridcolor='rgba(139, 92, 246, 0.2)')
        st.plotly_chart(fig, use_container_width=True)

    # Confusion Matrix
    st.markdown("### Confusion Matrix")

    cm = results[selected_model]['confusion_matrix']

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted: No Churn', 'Predicted: Churn'],
        y=['Actual: No Churn', 'Actual: Churn'],
        colorscale=[[0, '#1e1e3f'], [1, COLORS['primary']]],
        text=cm,
        texttemplate='%{text}',
        textfont={'size': 20, 'color': 'white'},
        hoverinfo='z'
    ))

    fig.update_layout(
        title=f'Confusion Matrix - {selected_model}',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Model-specific insights
    st.markdown("### Model Analysis")

    model_insights = {
        'Logistic Regression': """
        <div class='info-box'>
            <h4>Why Logistic Regression Performs Well</h4>
            <ul>
                <li><strong>Accuracy: ~81%</strong> | <strong>ROC AUC: 0.84</strong></li>
                <li>Good at identifying customers who will <em>not</em> churn (high precision for class 0)</li>
                <li>Catches about 56% of actual churners. Not ideal, but acceptable given the class imbalance.</li>
                <li><strong>Key Limitation:</strong> Assumes a single global relationship between features and churn.
                An increase in monthly payments affects all customers equally in this model.</li>
                <li><strong>Strength:</strong> Highly interpretable. We can look at the coefficients to understand
                feature impact, which is great for explaining predictions to stakeholders.</li>
            </ul>
        </div>
        """,
        'K-Nearest Neighbors': """
        <div class='info-box'>
            <h4>Why KNN Underperforms</h4>
            <ul>
                <li><strong>Accuracy: ~77%</strong> | <strong>ROC AUC: 0.79</strong></li>
                <li>Here's the thing: just always predicting "no churn" would give 73.5% accuracy. KNN only beats that by about 3.5%.</li>
                <li><strong>Key Issues:</strong>
                    <ul>
                        <li><strong>Class Imbalance:</strong> With more non-churners in the dataset, neighbors of a
                        new point are more likely to be non-churners, which biases predictions.</li>
                        <li><strong>High Dimensionality:</strong> With 19 features, distance metrics become less
                        meaningful (the curse of dimensionality).</li>
                        <li><strong>No Learning:</strong> KNN doesn't actually learn anything. It just memorizes and
                        compares distances, making it sensitive to noise and irrelevant features.</li>
                    </ul>
                </li>
                <li><strong>Could Improve By:</strong> Feature selection, dimensionality reduction (PCA), or SMOTE for class balancing.</li>
            </ul>
        </div>
        """,
        'Decision Tree': """
        <div class='info-box'>
            <h4>Decision Tree: The Hidden Insight</h4>
            <ul>
                <li><strong>Accuracy: ~76%</strong> | <strong>ROC AUC: 0.79</strong></li>
                <li>Without depth limiting, we got 93% train accuracy but only 72% test accuracy. That's <strong>severe overfitting</strong>.</li>
                <li>With max_depth=7, we get better generalization but lower accuracy than ensemble methods.</li>
                <li><strong>Critical Discovery:</strong> The root node splits on <strong>Fiber Optic Internet</strong>.
                This confirms our EDA finding and reveals something important:</li>
                <li style='color: #a78bfa;'><em>Different customer segments behave differently.</em> Fiber optic customers
                have different churn patterns than DSL customers. This is <strong>non-linear behavior</strong> that
                logistic regression's single global function can't capture.</li>
                <li><strong>Business Insight:</strong> We should analyze fiber optic and DSL customers separately
                to understand why they churn differently.</li>
            </ul>
        </div>
        """,
        'Random Forest': """
        <div class='info-box'>
            <h4>Why Random Forest is Recommended</h4>
            <ul>
                <li><strong>Accuracy: ~80%</strong> | <strong>ROC AUC: 0.84</strong></li>
                <li>Matches Logistic Regression's AUC while capturing <strong>non-linear patterns</strong></li>
                <li>Good generalization. Train accuracy (82%) is close to test accuracy (80%), meaning low overfitting.</li>
                <li><strong>Key Advantages:</strong>
                    <ul>
                        <li>Combines many decision trees, reducing variance and overfitting</li>
                        <li>Captures the insight that different customer groups behave differently</li>
                        <li>Provides feature importance rankings</li>
                    </ul>
                </li>
                <li><strong>Feature Importance Reveals:</strong> Tenure > Total Charges > Fiber Optic > Monthly Charges > Contract Type</li>
                <li><strong>Why Choose This:</strong> Our data is non-linear (different segments, different behaviors).
                Random Forest handles this better than a single global model like Logistic Regression.</li>
            </ul>
        </div>
        """
    }

    st.markdown(model_insights.get(selected_model, ""), unsafe_allow_html=True)

    # All models ROC comparison
    st.markdown("### All Models ROC Comparison")

    fig = go.Figure()
    colors = [COLORS['primary'], COLORS['success'], COLORS['warning'], COLORS['danger']]

    for (name, data), color in zip(results.items(), colors):
        fig.add_trace(go.Scatter(
            x=data['fpr'],
            y=data['tpr'],
            name=f'{name} (AUC={data["roc_auc"]:.3f})',
            line=dict(color=color, width=2)
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random',
        line=dict(color='gray', dash='dash')
    ))

    fig.update_layout(
        title='ROC Curves - All Models',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text']},
        height=450,
        legend=dict(x=0.6, y=0.1)
    )
    fig.update_xaxes(gridcolor='rgba(139, 92, 246, 0.2)')
    fig.update_yaxes(gridcolor='rgba(139, 92, 246, 0.2)')
    st.plotly_chart(fig, use_container_width=True)


def show_predictor(results, feature_cols, num_cols, cat_cols, df):
    """Interactive churn prediction."""
    st.markdown("<h1>Try It Yourself</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class='highlight-box'>
        <p>Enter customer details below to predict their likelihood of churning.
        The prediction uses our best-performing <strong>Random Forest</strong> model.</p>
    </div>
    """, unsafe_allow_html=True)

    # Input form
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Demographics")
        gender = st.selectbox("Gender", ['Male', 'Female'])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        partner = st.selectbox("Has Partner", ['Yes', 'No'])
        dependents = st.selectbox("Has Dependents", ['Yes', 'No'])

    with col2:
        st.markdown("#### Services")
        phone = st.selectbox("Phone Service", ['Yes', 'No'])
        multiple_lines = st.selectbox("Multiple Lines", ['No', 'Yes', 'No phone service'])
        internet = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
        security = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
        backup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
        protection = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
        support = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
        tv = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
        movies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])

    with col3:
        st.markdown("#### Billing")
        contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
        paperless = st.selectbox("Paperless Billing", ['Yes', 'No'])
        payment = st.selectbox("Payment Method", [
            'Electronic check', 'Mailed check',
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
        total = st.number_input("Total Charges ($)", min_value=0.0, value=monthly * tenure)

    # Predict button
    if st.button("Predict Churn Risk", use_container_width=True):
        # Create input dataframe
        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [senior],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet],
            'OnlineSecurity': [security],
            'OnlineBackup': [backup],
            'DeviceProtection': [protection],
            'TechSupport': [support],
            'StreamingTV': [tv],
            'StreamingMovies': [movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless],
            'PaymentMethod': [payment],
            'MonthlyCharges': [monthly],
            'TotalCharges': [total]
        })

        # Get prediction from Random Forest
        model = results['Random Forest']['pipeline']
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        churn_prob = probability[1] * 100
        retain_prob = probability[0] * 100

        st.markdown("---")
        st.markdown("### Prediction Results")

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            if prediction == 1:
                st.markdown(f"""
                <div style='
                    text-align: center;
                    padding: 30px;
                    background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%);
                    border: 2px solid {COLORS['danger']};
                    border-radius: 15px;
                    box-shadow: 0 4px 20px rgba(239, 68, 68, 0.3);
                '>
                    <h2 style='color: {COLORS['danger']}; margin: 0;'>HIGH CHURN RISK</h2>
                    <p style='font-size: 3rem; color: {COLORS['danger']}; margin: 10px 0;'>{churn_prob:.1f}%</p>
                    <p style='color: #a78bfa;'>Probability of Churning</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='
                    text-align: center;
                    padding: 30px;
                    background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%);
                    border: 2px solid {COLORS['success']};
                    border-radius: 15px;
                    box-shadow: 0 4px 20px rgba(16, 185, 129, 0.3);
                '>
                    <h2 style='color: {COLORS['success']}; margin: 0;'>LOW CHURN RISK</h2>
                    <p style='font-size: 3rem; color: {COLORS['success']}; margin: 10px 0;'>{retain_prob:.1f}%</p>
                    <p style='color: #a78bfa;'>Probability of Staying</p>
                </div>
                """, unsafe_allow_html=True)

        # Risk factors
        st.markdown("### Risk Factor Analysis")

        risk_factors = []
        if contract == 'Month-to-month':
            risk_factors.append("Month-to-month contract (higher churn risk)")
        if payment == 'Electronic check':
            risk_factors.append("Electronic check payment (associated with higher churn)")
        if internet == 'Fiber optic':
            risk_factors.append("Fiber optic internet (higher churn rate observed)")
        if tenure < 12:
            risk_factors.append(f"Short tenure ({tenure} months) - new customers churn more")
        if monthly > 70:
            risk_factors.append(f"High monthly charges (${monthly:.2f})")

        protective_factors = []
        if contract in ['One year', 'Two year']:
            protective_factors.append(f"{contract} contract (lower churn risk)")
        if tenure > 24:
            protective_factors.append(f"Long tenure ({tenure} months) - loyal customer")
        if security == 'Yes' or support == 'Yes':
            protective_factors.append("Has additional services (security/support)")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Risk Factors:**")
            if risk_factors:
                for factor in risk_factors:
                    st.markdown(f"- {factor}")
            else:
                st.markdown("- No significant risk factors identified")

        with col2:
            st.markdown("**Protective Factors:**")
            if protective_factors:
                for factor in protective_factors:
                    st.markdown(f"- {factor}")
            else:
                st.markdown("- No protective factors identified")


def show_findings(results):
    """Display key findings and recommendations."""
    st.markdown("<h1>Takeaways</h1>", unsafe_allow_html=True)

    # Model comparison summary
    st.markdown("### How Each Model Performed")

    model_data = []
    for name, data in results.items():
        model_data.append({
            'Model': name,
            'Accuracy': f"{data['accuracy']*100:.1f}%",
            'ROC AUC': f"{data['roc_auc']:.3f}",
            'Strengths': 'Interpretable coefficients' if name == 'Logistic Regression' else
                        'Simple, no training' if name == 'K-Nearest Neighbors' else
                        'Reveals customer segments' if name == 'Decision Tree' else 'Best overall + feature importance',
            'Limitations': 'Assumes linear relationships' if name == 'Logistic Regression' else
                          'Struggles with imbalance & high dimensions' if name == 'K-Nearest Neighbors' else
                          'Prone to overfitting' if name == 'Decision Tree' else 'Less interpretable than LR'
        })

    st.dataframe(pd.DataFrame(model_data), use_container_width=True, hide_index=True)

    # The big discovery section
    st.markdown("### The Big Discovery")

    st.markdown("""
    <div class='highlight-box'>
        <p>While Logistic Regression and Random Forest achieved similar ROC AUC scores (~0.84), the
        <strong>Decision Tree revealed something crucial</strong> that changes how we should think about this problem:</p>
        <br>
        <h4 style='color: #8b5cf6;'>Our data is NOT linear.</h4>
        <br>
        <p>The Decision Tree's root node splits on <strong>Fiber Optic Internet</strong>, showing that
        fiber optic customers behave fundamentally differently from DSL/no-internet customers.</p>
        <br>
        <p><strong>What this means:</strong></p>
        <ul>
            <li>Logistic Regression assumes that an increase in monthly payments affects churn probability
            <em>equally for all customers</em>. But that's not how it works in reality.</li>
            <li>A $10 increase in monthly charges might significantly increase churn risk for a new fiber optic
            customer, but barely affect a long-tenured DSL customer.</li>
            <li>We have <strong>different customer segments with different behaviors</strong>.</li>
        </ul>
        <br>
        <p><strong>Recommendation:</strong> Use Random Forest for predictions (captures non-linearity), but
        analyze customer segments separately for targeted retention strategies.</p>
    </div>
    """, unsafe_allow_html=True)

    # Key findings - full width layout to break symmetry
    st.markdown("### What Stood Out")

    st.markdown("""
    <div class='highlight-box'>
        <h4>Churn Predictors (Strongest to Weakest)</h4>
        <p><em>Based on Random Forest feature importance:</em></p>
        <div style='display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px;'>
            <span style='background: rgba(139, 92, 246, 0.2); padding: 8px 15px; border-radius: 20px; border: 1px solid #6c4ab6;'><strong>Tenure (21.6%)</strong> - New customers are high risk</span>
            <span style='background: rgba(139, 92, 246, 0.2); padding: 8px 15px; border-radius: 20px; border: 1px solid #6c4ab6;'><strong>Total Charges (13.5%)</strong> - Long-term commitment matters</span>
            <span style='background: rgba(139, 92, 246, 0.2); padding: 8px 15px; border-radius: 20px; border: 1px solid #6c4ab6;'><strong>Fiber Optic (9.3%)</strong> - Unique churn patterns</span>
            <span style='background: rgba(139, 92, 246, 0.2); padding: 8px 15px; border-radius: 20px; border: 1px solid #6c4ab6;'><strong>Monthly Charges (8.5%)</strong> - Higher bills = higher churn</span>
            <span style='background: rgba(139, 92, 246, 0.2); padding: 8px 15px; border-radius: 20px; border: 1px solid #6c4ab6;'><strong>Two-Year Contract (7.9%)</strong> - Reduces churn dramatically</span>
            <span style='background: rgba(139, 92, 246, 0.2); padding: 8px 15px; border-radius: 20px; border: 1px solid #6c4ab6;'><strong>Electronic Check (7.7%)</strong> - Red flag payment method</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box' style='margin-top: 15px;'>
        <h4>Technical Findings</h4>
        <p><strong>Class Imbalance (73.5% vs 26.5%):</strong> All models struggle to detect churners.
        Precision is high for non-churners (~85%) but lower for churners (~56-67%). This is expected
        and why we prioritize ROC AUC over raw accuracy.</p>
        <p style='margin-top: 10px;'><strong>Non-linear Relationships:</strong> Different customer segments respond differently
        to the same features. One-size-fits-all models miss this nuance.</p>
        <p style='margin-top: 10px;'><strong>Feature Interactions:</strong> High monthly charges + low tenure + month-to-month
        contract + electronic check = highest risk profile.</p>
        <p style='margin-top: 10px;'><strong>Model Selection:</strong> Random Forest is recommended. It has similar accuracy to Logistic
        Regression but captures the non-linear segment behaviors that the Decision Tree analysis revealed.</p>
    </div>
    """, unsafe_allow_html=True)

    # Business recommendations
    st.markdown("### Business Recommendations")

    recommendations = [
        {
            "title": "Target New Customers",
            "desc": "Implement retention programs for customers in their first 12 months. This is the critical period."
        },
        {
            "title": "Incentivize Long-term Contracts",
            "desc": "Offer discounts or perks for customers who sign one or two-year contracts."
        },
        {
            "title": "Payment Method Analysis",
            "desc": "Investigate why electronic check users churn more. Consider offering autopay incentives."
        },
        {
            "title": "Fiber Optic Service Review",
            "desc": "Review fiber optic service quality and pricing. High churn suggests customer dissatisfaction."
        },
        {
            "title": "Bundle Services",
            "desc": "Customers with more services stay longer. Create attractive service bundles."
        },
        {
            "title": "Early Warning System",
            "desc": "Deploy this model to identify at-risk customers and trigger proactive retention outreach."
        }
    ]

    cols = st.columns(3)
    for i, rec in enumerate(recommendations):
        with cols[i % 3]:
            st.markdown(f"""
            <div style='
                padding: 20px;
                background: linear-gradient(135deg, #2d1b4e 0%, #1e1e3f 100%);
                border: 1px solid #6c4ab6;
                border-radius: 10px;
                margin-bottom: 15px;
                min-height: 130px;
            '>
                <h4 style='color: #a78bfa; margin: 0;'>{rec['title']}</h4>
                <p style='color: #e2e8f0; font-size: 0.9rem; margin-top: 10px;'>{rec['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

    # Why each model behaved as it did
    st.markdown("### Why Each Model Behaved the Way It Did")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='info-box'>
            <h4>Why Logistic Regression Works Well</h4>
            <p>Despite assuming linear relationships, LR achieves 0.84 AUC because the <em>overall trends</em>
            are captured: higher tenure = lower churn, higher charges = higher churn. It's interpretable
            and gives us coefficient insights.</p>
            <br>
            <h4>Why KNN Struggles</h4>
            <p>KNN doesn't actually learn anything. It just memorizes. With 73.5% non-churners, most "neighbors" of any point are
            going to be non-churners. Add in 19 features (high dimensionality) where distance becomes less meaningful,
            and KNN can barely beat the baseline of just always predicting "no churn."</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='info-box'>
            <h4>Why Decision Tree Reveals Segments</h4>
            <p>Trees split data recursively. The first split on Fiber Optic shows that this single feature
            creates two fundamentally different populations. Within each population, different rules apply.
            This is non-linearity that LR misses.</p>
            <br>
            <h4>Why Random Forest is Best</h4>
            <p>RF combines many trees, each seeing different data subsets and features. This reduces
            overfitting while preserving the ability to capture non-linear segment behaviors. It matches
            LR's AUC while being more flexible.</p>
        </div>
        """, unsafe_allow_html=True)

    # Personal Reflection
    st.markdown("### What I Learned")

    st.markdown("""
    <div class='highlight-box'>
        <p>One interesting thing I learned from this project is that customers don't always behave the way you think they will.
        That's why it's important to analyze data and start asking questions.</p>
        <br>
        <p>For example, in this dataset the majority of customers who paid by electronic check churned. You have to ask yourself why.
        They could have potentially loved the services, but the inconvenience of paying made them leave. This is important because
        if they had an easier method of payment they could have stayed, and that's good for business.</p>
        <br>
        <p>On the machine learning side, high class imbalance can lead to a biased model. When one class dominates the dataset,
        the model learns to favor that class because it's the "safe" prediction. In our case, predicting "no churn" for everyone
        would still give 73.5% accuracy. That's why metrics like ROC AUC matter more here, since they measure how well the model
        distinguishes between classes, not just how often it's right.</p>
        <br>
        <p>As your data grows it can evolve, so it's important to analyze and check on your data regularly. Customer behavior changes
        over time, and what worked yesterday might not work tomorrow. Keep asking questions and keep exploring.</p>
    </div>
    """, unsafe_allow_html=True)

    # Wrapping Up
    st.markdown("""
    <div class='highlight-box' style='margin-top: 30px;'>
        <h3>Wrapping Up</h3>
        <p>This analysis shows the power of machine learning in predicting customer churn.
        Both <strong>Logistic Regression</strong> and <strong>Random Forest</strong> achieved solid performance
        with ~80% accuracy and 0.84 ROC AUC scores.</p>
        <br>
        <p>The big insight from the Decision Tree analysis is that <strong>customer behavior is non-linear</strong>.
        Different customer segments respond differently to the same factors. Logistic Regression assumes one global
        relationship for all customers, but reality is more nuanced. A fiber optic customer with 2 months tenure
        behaves very differently from a DSL customer with 3 years tenure, even if they have similar monthly charges.</p>
        <br>
        <p>This suggests that <strong>personalized retention strategies</strong> would be more effective than
        one-size-fits-all approaches. Segment customers by internet type and tenure, then apply targeted interventions.</p>
        <br>
        <p><em><strong>My Recommendation:</strong> Deploy Random Forest for production predictions since it captures
        non-linear patterns well. Use Logistic Regression coefficients when you need to explain things to stakeholders.
        Look into fiber optic service quality and target new, high-paying customers with early retention incentives.</em></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
