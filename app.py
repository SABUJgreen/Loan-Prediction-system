import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys

# Ensure current directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

import ml_engine

# --- Page Configuration ---
st.set_page_config(
    page_title="LoanGuard AI | Credit Intelligence Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Header Banner */
    .hero-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        color: white;
        padding: 1.8rem 2.2rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        color: #ffffff;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-top: 0.35rem;
        margin-bottom: 0;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.95rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 0.3rem;
        line-height: 1.1;
    }
    .metric-delta {
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 0.4rem;
    }
    .delta-positive { color: #059669; }
    .delta-neutral { color: #2563eb; }
    
    /* Decision Cards */
    .decision-box-approved {
        background: linear-gradient(135deg, #064e3b 0%, #047857 100%);
        color: #ffffff;
        padding: 1.8rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(5, 150, 105, 0.35);
    }
    .decision-box-rejected {
        background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 100%);
        color: #ffffff;
        padding: 1.8rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(185, 28, 28, 0.35);
    }
    
    /* Tags and Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .badge-approved { background-color: #d1fae5; color: #065f46; }
    .badge-rejected { background-color: #fee2e2; color: #991b1b; }
    .badge-prime { background-color: #dbeafe; color: #1e40af; }
    
    /* Factor list */
    .factor-item {
        background: #f8fafc;
        border-left: 4px solid #0ea5e9;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
        font-size: 0.92rem;
        color: #1e293b;
    }
    .factor-item.warning {
        border-left-color: #f59e0b;
        background: #fffbeb;
    }
    
    /* Clean Streamlit elements */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Cached Model & Data Loading ---
@st.cache_resource(show_spinner="Training machine learning ensemble...")
def get_cached_ml_data():
    return ml_engine.train_and_evaluate_all()

data_pack = get_cached_ml_data()
results = data_pack['results']
leaderboard = data_pack['leaderboard']
raw_train = data_pack['raw_train']
raw_test = data_pack['raw_test']
clean_train = data_pack['clean_train']

# --- Top Navigation / Hero Banner ---
st.markdown("""
<div class="hero-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1 class="hero-title">💳 LoanGuard AI • Credit Decision Platform</h1>
            <p class="hero-subtitle">End-to-End Automated Loan Underwriting, Risk Analytics & Predictive Modeling</p>
        </div>
        <div style="margin-top: 0.5rem;">
            <span class="badge" style="background: rgba(255,255,255,0.15); color: #ffffff; border: 1px solid rgba(255,255,255,0.25);">
                Ensemble: 8 ML Algorithms
            </span>
            &nbsp;
            <span class="badge" style="background: #10b981; color: white;">
                Active: 10-Fold Cross Validated
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Controls ---
with st.sidebar:
    st.image("https://img.icons8.com/isometric-line/100/bank-building.png", width=64)
    st.title("LoanGuard Control")
    st.caption("Automated Loan Underwriting Engine")
    st.markdown("---")
    
    menu = st.radio(
        "Navigation",
        [
            "📊 Executive Overview & KPIs",
            "🎯 Instant Loan Simulator",
            "📈 Exploratory Data Analytics (EDA)",
            "⚖️ Model Benchmark Arena",
            "📂 Batch Scoring & Data Explorer"
        ],
        index=0
    )
    
    st.markdown("---")
    st.subheader("⚙️ Active Decision Model")
    selected_model_name = st.selectbox(
        "Classifier for Simulation",
        options=list(results.keys()),
        index=0,
        help="Choose the algorithm that evaluates individual loans in the simulator."
    )
    
    active_model_stats = results[selected_model_name]
    st.info(f"**{selected_model_name}**\n- CV Accuracy: **{active_model_stats['cv_mean']*100:.2f}%**\n- ROC AUC: **{active_model_stats['roc_auc']:.3f}**")
    
    st.markdown("---")
    st.caption("Loan Prediction System v2.0 • Antigravity Edition")
    st.caption("Data source: Kaggle / Analytics Vidhya Loan Prediction Dataset")

# ==============================================================================
# TAB 1: EXECUTIVE OVERVIEW & KPIS
# ==============================================================================
if menu == "📊 Executive Overview & KPIs":
    st.subheader("Executive Underwriting Portfolio Overview")
    st.write("Real-time portfolio metrics, loan approval distribution, and macroeconomic risk indicators.")
    
    # Key Financial Metrics Row
    total_apps = len(raw_train)
    approved_apps = (raw_train['Loan_Status'] == 'Y').sum()
    approval_rate = (approved_apps / total_apps) * 100
    avg_loan = raw_train['LoanAmount'].mean() * 1000
    median_income = raw_train['ApplicantIncome'].median()
    avg_household = (raw_train['ApplicantIncome'] + raw_train['CoapplicantIncome']).mean()
    
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Historical Apps</div>
            <div class="metric-value">{total_apps}</div>
            <div class="metric-delta delta-neutral">Train Dataset</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Approval Rate</div>
            <div class="metric-value">{approval_rate:.1f}%</div>
            <div class="metric-delta delta-positive">{approved_apps} Approved / {total_apps - approved_apps} Rejected</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Loan Request</div>
            <div class="metric-value">${avg_loan:,.0f}</div>
            <div class="metric-delta delta-neutral">Median: ${raw_train['LoanAmount'].median()*1000:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Median App Income</div>
            <div class="metric-value">${median_income:,.0f}</div>
            <div class="metric-delta delta-neutral">Monthly Base</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Household Inc</div>
            <div class="metric-value">${avg_household:,.0f}</div>
            <div class="metric-delta delta-positive">Applicant + Co-applicant</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts Grid
    row1_c1, row1_c2, row1_c3 = st.columns([1, 1, 1])
    
    with row1_c1:
        # Approval Donut
        status_counts = raw_train['Loan_Status'].map({'Y': 'Approved (Y)', 'N': 'Rejected (N)'}).value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        fig_donut = px.pie(
            status_counts, 
            values='Count', 
            names='Status', 
            hole=0.6,
            title="Loan Status Distribution",
            color='Status',
            color_discrete_map={'Approved (Y)': '#10b981', 'Rejected (N)': '#ef4444'}
        )
        fig_donut.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=300, showlegend=True)
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with row1_c2:
        # Approval by Credit History
        df_ch = raw_train.copy()
        df_ch['Credit_Label'] = df_ch['Credit_History'].map({1.0: 'Has Credit History (1.0)', 0.0: 'No Credit History (0.0)'}).fillna('Missing')
        ch_grp = df_ch.groupby(['Credit_Label', 'Loan_Status']).size().unstack(fill_value=0)
        ch_pct = (ch_grp.get('Y', 0) / ch_grp.sum(axis=1) * 100).reset_index()
        ch_pct.columns = ['Credit History', 'Approval Rate (%)']
        
        fig_ch = px.bar(
            ch_pct,
            x='Credit History',
            y='Approval Rate (%)',
            title="Approval Rate by Credit History",
            text_auto='.1f',
            color='Approval Rate (%)',
            color_continuous_scale=['#f87171', '#34d399']
        )
        fig_ch.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=300, yaxis_range=[0, 100], coloraxis_showscale=False)
        st.plotly_chart(fig_ch, use_container_width=True)
        
    with row1_c3:
        # Approval by Property Area
        pa_grp = raw_train.groupby(['Property_Area', 'Loan_Status']).size().unstack(fill_value=0)
        pa_pct = (pa_grp['Y'] / pa_grp.sum(axis=1) * 100).reset_index()
        pa_pct.columns = ['Property Area', 'Approval Rate (%)']
        
        fig_pa = px.bar(
            pa_pct,
            x='Property Area',
            y='Approval Rate (%)',
            title="Approval Rate by Property Region",
            text_auto='.1f',
            color='Property Area',
            color_discrete_sequence=['#3b82f6', '#10b981', '#f59e0b']
        )
        fig_pa.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=300, yaxis_range=[0, 100], showlegend=False)
        st.plotly_chart(fig_pa, use_container_width=True)
        
    st.markdown("### 📌 Executive Insights & Credit Policy Findings")
    col_i1, col_i2, col_i3 = st.columns(3)
    with col_i1:
        st.markdown("""
        <div class="factor-item">
            <strong>Credit History is Decisive (79.6% vs 8.0%)</strong><br>
            Applicants with verified credit history are almost 10x more likely to secure loan approval. It represents over 70% of feature weight across tree and linear models.
        </div>
        """, unsafe_allow_html=True)
    with col_i2:
        st.markdown("""
        <div class="factor-item">
            <strong>Semiurban Market Outperformance</strong><br>
            Semiurban properties exhibit the highest approval conversion at <strong>76.8%</strong>, compared to 65.8% in Urban regions and 61.5% in Rural regions.
        </div>
        """, unsafe_allow_html=True)
    with col_i3:
        st.markdown("""
        <div class="factor-item">
            <strong>Co-Applicant Leverage Boosts Pass Rates</strong><br>
            Applications featuring a co-applicant income exceed solitary borrower approvals by 14.2%, substantially dampening the Debt-To-Income risk ratio.
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# TAB 2: INSTANT LOAN ELIGIBILITY SIMULATOR
# ==============================================================================
elif menu == "🎯 Instant Loan Simulator":
    st.subheader("Real-Time Loan Decision & Risk Simulator")
    st.write("Input applicant demographic, financial, and property credentials to generate an automated underwriting verdict.")
    
    col_input, col_output = st.columns([1.1, 1])
    
    with col_input:
        st.markdown("#### 1. Applicant Profile")
        f_c1, f_c2, f_c3 = st.columns(3)
        with f_c1:
            gender = st.selectbox("Gender", ["Male", "Female"], index=0)
            married = st.selectbox("Marital Status", ["Yes", "No"], index=0)
        with f_c2:
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], index=0)
            education = st.selectbox("Education", ["Graduate", "Not Graduate"], index=0)
        with f_c3:
            self_employed = st.selectbox("Self Employed", ["No", "Yes"], index=0)
            property_area = st.selectbox("Property Area", ["Semiurban", "Urban", "Rural"], index=0)
            
        st.markdown("#### 2. Financial & Credit Credentials")
        f_c4, f_c5 = st.columns(2)
        with f_c4:
            applicant_income = st.number_input("Applicant Monthly Income ($)", min_value=0, max_value=150000, value=5000, step=500)
            coapplicant_income = st.number_input("Co-Applicant Monthly Income ($)", min_value=0, max_value=80000, value=1800, step=250)
            credit_history = st.selectbox(
                "Credit History Records",
                options=[1.0, 0.0],
                format_func=lambda x: "Meets Credit Guidelines (1.0 - Pass)" if x == 1.0 else "Delinquent / No History (0.0 - Fail)",
                index=0
            )
        with f_c5:
            loan_amount_k = st.number_input("Loan Amount Requested ($ in Thousands)", min_value=10, max_value=1000, value=140, step=10, help="e.g. 140 = $140,000")
            loan_term_months = st.selectbox(
                "Loan Term (Months)",
                options=[12.0, 36.0, 60.0, 84.0, 120.0, 180.0, 240.0, 300.0, 360.0, 480.0],
                index=8, # 360 months (30 years)
                format_func=lambda x: f"{int(x)} months ({int(x/12)} years)"
            )
            interest_rate = st.slider("Indicative Annual Interest Rate (%)", min_value=4.0, max_value=15.0, value=8.5, step=0.25)

        evaluate_btn = st.button("🚀 Evaluate Loan Application", type="primary", use_container_width=True)
        
    with col_output:
        st.markdown("#### 3. Real-Time Underwriting Assessment")
        
        applicant_payload = {
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount_k,
            'Loan_Amount_Term': loan_term_months,
            'Credit_History': credit_history,
            'Property_Area': property_area
        }
        
        active_model = results[selected_model_name]['model']
        decision = ml_engine.predict_single_applicant(active_model, applicant_payload)
        
        prob = decision['probability']
        is_approved = decision['prediction'] == 1
        
        # Financial Health Metrics preview
        total_income = decision['total_income']
        emi = decision['emi']
        emi_ratio = decision['emi_ratio']
        
        m_c1, m_c2, m_c3 = st.columns(3)
        with m_c1:
            st.metric("Total Household Income", f"${total_income:,.0f}/mo")
        with m_c2:
            st.metric("Est. Monthly EMI", f"${emi:,.0f}/mo")
        with m_c3:
            st.metric("Debt-to-Income (EMI Ratio)", f"{emi_ratio:.1f}%", delta="- Safe (<35%)" if emi_ratio <= 35 else "- Strained" if emi_ratio <= 50 else "- Critical (>50%)", delta_color="normal" if emi_ratio <= 35 else "inverse")

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Decision Banner
        if is_approved:
            st.markdown(f"""
            <div class="decision-box-approved">
                <h2 style="margin:0; font-size: 2.2rem; font-weight: 800;">✅ LOAN APPROVED</h2>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.95;">
                    Approval Probability: <strong>{prob*100:.1f}%</strong> | Risk Rating: <strong>{decision['risk_tier']}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="decision-box-rejected">
                <h2 style="margin:0; font-size: 2.2rem; font-weight: 800;">❌ APPLICATION DECLINED</h2>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.95;">
                    Approval Probability: <strong>{prob*100:.1f}%</strong> | Risk Rating: <strong>{decision['risk_tier']}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Confidence Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Model Approval Confidence (%)", 'font': {'size': 16}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#10b981" if is_approved else "#ef4444"},
                'steps': [
                    {'range': [0, 45], 'color': "rgba(239, 68, 68, 0.15)"},
                    {'range': [45, 70], 'color': "rgba(245, 158, 11, 0.15)"},
                    {'range': [70, 100], 'color': "rgba(16, 185, 129, 0.15)"}
                ],
                'threshold': {
                    'line': {'color': "#0f172a", 'width': 3},
                    'thickness': 0.8,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(height=180, margin=dict(t=30, b=10, l=30, r=30))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Explanatory factors
        st.markdown("##### 🔍 Underwriting Factors & Audit")
        for pos in decision['positives']:
            st.markdown(f'<div class="factor-item">✅ {pos}</div>', unsafe_allow_html=True)
        for cau in decision['cautions']:
            st.markdown(f'<div class="factor-item warning">⚠️ {cau}</div>', unsafe_allow_html=True)
            
        # Download summary report
        summary_payload = {
            "Applicant": applicant_payload,
            "Decision": {
                "Status": decision['status_label'],
                "Approval_Probability": f"{prob*100:.2f}%",
                "Risk_Tier": decision['risk_tier'],
                "Classifier_Model": selected_model_name,
                "Estimated_Monthly_EMI": f"${emi:.2f}",
                "Household_Income": f"${total_income:.2f}",
                "Debt_To_Income_Ratio": f"{emi_ratio:.2f}%"
            }
        }
        st.download_button(
            "📥 Download Underwriting Assessment Report (JSON)",
            data=json.dumps(summary_payload, indent=2),
            file_name="Loan_Assessment_Decision.json",
            mime="application/json",
            use_container_width=True
        )

# ==============================================================================
# TAB 3: EXPLORATORY DATA ANALYTICS (EDA)
# ==============================================================================
elif menu == "📈 Exploratory Data Analytics (EDA)":
    st.subheader("Exploratory Data Analysis & Feature Intersections")
    st.write("Deep dive into demographic correlations, financial distributions, and borrower behaviors.")
    
    eda_tab1, eda_tab2, eda_tab3 = st.tabs(["🔥 Correlation Matrix", "💰 Income vs Loan Dynamics", "👥 Demographics Breakdown"])
    
    with eda_tab1:
        st.markdown("#### Feature Correlation Heatmap")
        st.caption("Pearson correlation between numerical and encoded features in the training dataset.")
        
        corr_matrix = clean_train.drop(columns=['Loan_ID'], errors='ignore').corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale="RdBu_r",
            labels=dict(color="Correlation"),
            zmin=-1, zmax=1
        )
        fig_corr.update_layout(height=600, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with eda_tab2:
        col_s1, col_s2 = st.columns([1.3, 1])
        with col_s1:
            st.markdown("#### Applicant Income vs Loan Amount")
            clean_with_labels = clean_train.copy()
            clean_with_labels['Status'] = clean_with_labels['Loan_Status'].map({1: 'Approved (Y)', 0: 'Rejected (N)'})
            
            # Interactive Scatter
            fig_scatter = px.scatter(
                clean_with_labels,
                x='ApplicantIncome',
                y='LoanAmount',
                color='Status',
                size='CoapplicantIncome',
                hover_data=['Credit_History', 'Property_Area', 'Education'],
                color_discrete_map={'Approved (Y)': '#10b981', 'Rejected (N)': '#ef4444'},
                title="Applicant Income vs Loan Amount (Size = Coapplicant Income)",
                opacity=0.75
            )
            fig_scatter.update_layout(height=450)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with col_s2:
            st.markdown("#### Loan Amount Term Distribution")
            term_counts = raw_train['Loan_Amount_Term'].value_counts().reset_index()
            term_counts.columns = ['Term (Months)', 'Count']
            term_counts['Term (Months)'] = term_counts['Term (Months)'].astype(str)
            
            fig_term = px.bar(
                term_counts,
                x='Term (Months)',
                y='Count',
                title="Loan Repayment Term Distribution",
                text_auto=True,
                color='Count',
                color_continuous_scale='Blues'
            )
            fig_term.update_layout(height=450, coloraxis_showscale=False)
            st.plotly_chart(fig_term, use_container_width=True)
            
    with eda_tab3:
        d_c1, d_c2 = st.columns(2)
        with d_c1:
            st.markdown("#### Education vs Loan Status")
            edu_df = raw_train.groupby(['Education', 'Loan_Status']).size().reset_index(name='Count')
            edu_df['Status'] = edu_df['Loan_Status'].map({'Y': 'Approved', 'N': 'Rejected'})
            fig_edu = px.bar(
                edu_df,
                x='Education',
                y='Count',
                color='Status',
                barmode='group',
                color_discrete_map={'Approved': '#10b981', 'Rejected': '#ef4444'}
            )
            fig_edu.update_layout(height=380)
            st.plotly_chart(fig_edu, use_container_width=True)
            
        with d_c2:
            st.markdown("#### Dependents vs Loan Status")
            dep_df = raw_train.groupby(['Dependents', 'Loan_Status']).size().reset_index(name='Count')
            dep_df['Status'] = dep_df['Loan_Status'].map({'Y': 'Approved', 'N': 'Rejected'})
            fig_dep = px.bar(
                dep_df,
                x='Dependents',
                y='Count',
                color='Status',
                barmode='group',
                color_discrete_map={'Approved': '#10b981', 'Rejected': '#ef4444'}
            )
            fig_dep.update_layout(height=380)
            st.plotly_chart(fig_dep, use_container_width=True)

# ==============================================================================
# TAB 4: MODEL BENCHMARK & EVALUATION ARENA
# ==============================================================================
elif menu == "⚖️ Model Benchmark Arena":
    st.subheader("Machine Learning Algorithm Benchmark")
    st.write("Rigorous 10-Fold Stratified Cross-Validation and holdout testing comparing 8 classification architectures.")
    
    # Leaderboard Table
    st.dataframe(
        leaderboard.drop(columns=['_cv_val', '_val_acc']),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Deep dive for specific model
    col_eval1, col_eval2 = st.columns([1, 1.1])
    
    with col_eval1:
        st.markdown(f"#### Confusion Matrix: {selected_model_name}")
        model_pack = results[selected_model_name]
        cm = np.array(model_pack['confusion_matrix'])
        
        # Heatmap of Confusion Matrix
        cm_labels = [['True Negative (TN)', 'False Positive (FP)'],
                     ['False Negative (FN)', 'True Positive (TP)']]
        annotations_text = [[f"{cm[0][0]}<br>(TN)", f"{cm[0][1]}<br>(FP)"],
                            [f"{cm[1][0]}<br>(FN)", f"{cm[1][1]}<br>(TP)"]]
        
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Decision", y="Actual Truth", color="Cases"),
            x=['Rejected (0)', 'Approved (1)'],
            y=['Rejected (0)', 'Approved (1)'],
            text_auto=False,
            color_continuous_scale="Purples"
        )
        for i in range(2):
            for j in range(2):
                fig_cm.add_annotation(
                    x=j, y=i,
                    text=annotations_text[i][j],
                    showarrow=False,
                    font=dict(color="white" if cm[i][j] > cm.mean() else "black", size=14)
                )
        fig_cm.update_layout(height=380, margin=dict(t=30, b=20, l=20, r=20), coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)
        
        acc_pct = model_pack['test_acc'] * 100
        prec_pct = model_pack['precision'] * 100
        rec_pct = model_pack['recall'] * 100
        f1_sc = model_pack['f1']
        st.markdown(f"""
        **Holdout Validation Metrics**:
        - Accuracy: **{acc_pct:.2f}%**
        - Precision (Approved): **{prec_pct:.2f}%**
        - Recall (Sensitivity): **{rec_pct:.2f}%**
        - F1 Score: **{f1_sc:.3f}**
        """)
        
    with col_eval2:
        st.markdown(f"#### Feature Importances / Weights ({selected_model_name})")
        imp_df, metric_type = ml_engine.get_feature_importance(data_pack, selected_model_name)
        
        fig_imp = px.bar(
            imp_df.sort_values(by='Absolute_Impact', ascending=True),
            x='Absolute_Impact',
            y='Feature',
            orientation='h',
            title=f"Relative Feature Contribution ({metric_type})",
            color='Absolute_Impact',
            color_continuous_scale="Teal"
        )
        fig_imp.update_layout(height=420, margin=dict(t=40, b=20, l=20, r=20), coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)
        
    # Interactive Threshold Tuning Simulator
    st.markdown("---")
    st.subheader("🎚️ Decision Threshold Sensitivity Simulator")
    st.write("Tune the classification probability threshold to align with bank risk tolerance (Conservative vs Aggressive lending policy).")
    
    thresh = st.slider("Classification Threshold (Probability Cutoff)", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
    
    y_val = np.array(model_pack['y_val'])
    y_probs = np.array(model_pack['y_probs'])
    adjusted_preds = (y_probs >= thresh).astype(int)
    
    adj_acc = (adjusted_preds == y_val).mean() * 100
    adj_approvals = adjusted_preds.sum()
    total_val = len(y_val)
    adj_approval_pct = (adj_approvals / total_val) * 100
    
    th_c1, th_c2, th_c3 = st.columns(3)
    with th_c1:
        st.metric("Adjusted Accuracy", f"{adj_acc:.2f}%")
    with th_c2:
        st.metric("Total Approvals Passed", f"{adj_approvals} / {total_val}")
    with th_c3:
        st.metric("Portfolio Approval Rate", f"{adj_approval_pct:.1f}%")

# ==============================================================================
# TAB 5: BATCH SCORING & DATASET EXPLORER
# ==============================================================================
elif menu == "📂 Batch Scoring & Data Explorer":
    st.subheader("Batch Underwriting & Test Set Predictions")
    st.write("Review unlabeled production test applications, upload new borrower batches, and export decisioned CSVs.")
    
    batch_mode = st.radio("Choose Mode", ["Test Dataset (`Dataset/test.csv`)", "Upload Custom Batch CSV"], horizontal=True)
    
    if batch_mode == "Test Dataset (`Dataset/test.csv`)":
        st.markdown(f"#### Evaluated `test.csv` (367 Applications) via {selected_model_name}")
        
        # Run inference on test set
        clean_test = data_pack['clean_test']
        test_features = clean_test[ml_engine.FEATURE_COLS]
        active_clf = results[selected_model_name]['model']
        
        test_probs = active_clf.predict_proba(test_features)[:, 1] if hasattr(active_clf, "predict_proba") else active_clf.predict(test_features)
        test_preds = (test_probs >= 0.5).astype(int)
        
        scored_test = raw_test.copy()
        scored_test['Predicted_Status'] = np.where(test_preds == 1, 'Approved (Y)', 'Rejected (N)')
        scored_test['Confidence_Score'] = (test_probs * 100).round(2)
        scored_test['Risk_Tier'] = np.where(test_probs >= 0.75, 'Prime', np.where(test_probs >= 0.5, 'Near-Prime', 'High Risk'))
        
        # Batch summary metrics
        b_approved = (test_preds == 1).sum()
        b_rejected = (test_preds == 0).sum()
        
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.metric("Total Batch Records", len(scored_test))
        with b2:
            st.metric("Approved Applications", b_approved, delta=f"{(b_approved/len(scored_test))*100:.1f}%")
        with b3:
            st.metric("Rejected Applications", b_rejected, delta=f"{(b_rejected/len(scored_test))*100:.1f}%", delta_color="inverse")
        with b4:
            st.metric("Mean Approval Probability", f"{test_probs.mean()*100:.1f}%")
            
        # Filters
        st.markdown("##### Filter Applications")
        fil_c1, fil_c2, fil_c3 = st.columns(3)
        with fil_c1:
            status_filter = st.multiselect("Filter by Predicted Status", ['Approved (Y)', 'Rejected (N)'], default=['Approved (Y)', 'Rejected (N)'])
        with fil_c2:
            area_filter = st.multiselect("Filter by Property Area", scored_test['Property_Area'].unique().tolist(), default=scored_test['Property_Area'].unique().tolist())
        with fil_c3:
            search_id = st.text_input("Search Loan ID", "")
            
        filtered_df = scored_test[
            (scored_test['Predicted_Status'].isin(status_filter)) &
            (scored_test['Property_Area'].isin(area_filter))
        ]
        if search_id.strip():
            filtered_df = filtered_df[filtered_df['Loan_ID'].str.contains(search_id.strip(), case=False, na=False)]
            
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download output
        export_csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Export Scored Test Applications (CSV)",
            data=export_csv,
            file_name="Loan_Underwriting_Scored_Output.csv",
            mime="text/csv",
            type="primary"
        )
        
    else:
        st.markdown("#### Upload Custom Borrower CSV")
        st.write("Upload a CSV file with applicant attributes (`Gender`, `Married`, `ApplicantIncome`, `LoanAmount`, `Credit_History`, etc.).")
        
        uploaded_file = st.file_uploader("Upload Applicants CSV", type=["csv"])
        if uploaded_file is not None:
            user_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(user_df)} rows.")
            
            clean_user, _, _ = ml_engine.preprocess_data(user_df)
            user_features = clean_user[ml_engine.FEATURE_COLS]
            active_clf = results[selected_model_name]['model']
            
            user_probs = active_clf.predict_proba(user_features)[:, 1] if hasattr(active_clf, "predict_proba") else active_clf.predict(user_features)
            user_preds = (user_probs >= 0.5).astype(int)
            
            scored_user = user_df.copy()
            scored_user['Predicted_Status'] = np.where(user_preds == 1, 'Approved (Y)', 'Rejected (N)')
            scored_user['Approval_Probability (%)'] = (user_probs * 100).round(2)
            
            st.dataframe(scored_user, use_container_width=True)
            
            user_export = scored_user.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Scored Batch Results (CSV)",
                data=user_export,
                file_name="Uploaded_Applicants_Scored.csv",
                mime="text/csv"
            )
            
    st.markdown("---")
    st.markdown("#### Raw Training Data Explorer")
    with st.expander("🔍 Click to inspect raw historical training data (`Dataset/train.csv`)"):
        st.dataframe(raw_train, use_container_width=True)

