import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Want to Be an E-commerce: Predict Your Target Group",
    page_icon="🛍️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef2ff 0%, #f8fbff 45%, #ecfeff 100%);
    color: #1f2937;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

.hero-box {
    background: linear-gradient(90deg, #4f46e5 0%, #06b6d4 100%);
    padding: 2rem;
    border-radius: 22px;
    color: white;
    box-shadow: 0 10px 28px rgba(0,0,0,0.12);
    margin-bottom: 1.5rem;
}

.card {
    background: rgba(255, 255, 255, 0.92);
    padding: 1.4rem;
    border-radius: 20px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    margin-bottom: 1.2rem;
    border: 1px solid rgba(255,255,255,0.4);
}

.reco-card {
    background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
    padding: 1rem 1.2rem;
    border-radius: 18px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    text-align: center;
    margin-bottom: 1rem;
}

section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.82);
    backdrop-filter: blur(10px);
}

div[data-testid="stMetric"] {
    background-color: rgba(255,255,255,0.75);
    padding: 12px;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-box">
    <h1>🛍️ Want to Be an E-commerce: Predict Your Target Group</h1>
    <p style="font-size:17px; margin-bottom:0;">
        This interactive tool estimates <b>purchase success rate</b> and 
        <b>affordable spending capacity</b> based on customer profile data.  
        It also supports <b>multi-group comparison analysis</b> for target segmentation.
    </p>
</div>
""", unsafe_allow_html=True)

DATA_FILE = "acc102/output/high_frequency_user_profiles.csv"

@st.cache_data
def load_profile_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

df = load_profile_data(DATA_FILE)

def estimate_purchase_success_rate(monthly_income, monthly_expense, purchase_frequency, cancel_rate_pct):
    if monthly_income <= 0:
        return 0.0

    expense_ratio = monthly_expense / monthly_income
    base_score = 70
    freq_bonus = min(purchase_frequency * 3, 20)
    cancel_penalty = cancel_rate_pct * 0.5
    pressure_penalty = max(expense_ratio - 0.5, 0) * 40

    success_rate = base_score + freq_bonus - cancel_penalty - pressure_penalty
    success_rate = max(0, min(100, success_rate))
    return round(success_rate, 2)

def estimate_affordable_amount(monthly_income, monthly_expense, purchase_frequency, cancel_rate_pct):
    disposable_income = max(monthly_income - monthly_expense, 0)
    freq_factor = 1 + min(purchase_frequency / 20, 0.3)
    cancel_factor = 1 - min(cancel_rate_pct / 100, 0.5)
    affordable_amount = disposable_income * freq_factor * cancel_factor
    return round(max(0, affordable_amount), 2)

def risk_level(success_rate):
    if success_rate >= 80:
        return "Low Risk / High Success Probability"
    elif success_rate >= 60:
        return "Medium Risk / Relatively Stable"
    else:
        return "High Risk / Low Success Probability"

def recommend_price_range(affordable_amount):
    if affordable_amount <= 0:
        return {
            "recommended_range": "No recommended purchase range",
            "budget_range": "¥0 - ¥0",
            "mid_range": "Not recommended",
            "premium_range": "Not recommended",
            "advice": "Very limited affordable spending capacity."
        }
    elif affordable_amount < 500:
        return {
            "recommended_range": "Budget Range",
            "budget_range": f"¥0 - ¥{affordable_amount:,.0f}",
            "mid_range": "Above recommended budget",
            "premium_range": "Not recommended",
            "advice": "Focus on low-priced and entry-level products."
        }
    elif affordable_amount < 2000:
        return {
            "recommended_range": "Budget to Mid-range",
            "budget_range": f"¥0 - ¥{affordable_amount * 0.5:,.0f}",
            "mid_range": f"¥{affordable_amount * 0.5:,.0f} - ¥{affordable_amount:,.0f}",
            "premium_range": "Above recommended budget",
            "advice": "Mid-range products are acceptable, but premium pricing may reduce conversion."
        }
    else:
        return {
            "recommended_range": "Mid-range to Premium",
            "budget_range": f"¥0 - ¥{affordable_amount * 0.4:,.0f}",
            "mid_range": f"¥{affordable_amount * 0.4:,.0f} - ¥{affordable_amount * 0.8:,.0f}",
            "premium_range": f"¥{affordable_amount * 0.8:,.0f} - ¥{affordable_amount:,.0f}",
            "advice": "Suitable for higher-value products and premium offerings."
        }

def process_group_row(row):
    income = float(row["Monthly Income"])
    expense = float(row["Monthly Expense"])
    freq = float(row["Purchase Frequency"])
    cancel = float(row["Cancellation Rate (%)"])

    success = estimate_purchase_success_rate(income, expense, freq, cancel)
    affordable = estimate_affordable_amount(income, expense, freq, cancel)
    expense_ratio = round((expense / income) * 100, 2) if income > 0 else 0
    reco = recommend_price_range(affordable)
    risk = risk_level(success)

    return pd.Series({
        "Expense-to-Income Ratio (%)": expense_ratio,
        "Purchase Success Rate (%)": success,
        "Affordable Spending Amount": affordable,
        "Recommended Positioning": reco["recommended_range"],
        "Budget Range": reco["budget_range"],
        "Mid-range": reco["mid_range"],
        "Premium Range": reco["premium_range"],
        "Risk Level": risk
    })

tab1, tab2 = st.tabs(["Single Customer Prediction", "Multi-Group Comparison"])

with tab1:
    st.sidebar.header("Parameter Input")

    monthly_income = st.sidebar.number_input(
        "Monthly Income of Target Customer",
        min_value=0.0,
        value=8000.0,
        step=100.0
    )

    monthly_expense = st.sidebar.number_input(
        "Monthly Expense of Target Customer",
        min_value=0.0,
        value=3000.0,
        step=100.0
    )

    purchase_frequency = st.sidebar.number_input(
        "Monthly Purchase Frequency",
        min_value=0,
        value=5,
        step=1
    )

    cancel_rate_pct = st.sidebar.slider(
        "Order Cancellation Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=0.1
    )

   
    if df is not None:
        avg_expense = df["monthly_expense"].mean() if "monthly_expense" in df.columns else 0
        avg_freq = df["monthly_purchase_frequency"].mean() if "monthly_purchase_frequency" in df.columns else 0
        avg_cancel = df["cancel_rate_pct"].mean() if "cancel_rate_pct" in df.columns else 0
    else:
        avg_expense = avg_freq = avg_cancel = 0

    success_rate = estimate_purchase_success_rate(
        monthly_income, monthly_expense, purchase_frequency, cancel_rate_pct
    )
    affordable_amount = estimate_affordable_amount(
        monthly_income, monthly_expense, purchase_frequency, cancel_rate_pct
    )
    price_reco = recommend_price_range(affordable_amount)

    expense_ratio_pct = round((monthly_expense / monthly_income) * 100, 2) if monthly_income > 0 else 0
    remaining_income = max(monthly_income - monthly_expense, 0)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📈 Estimated Results")
        st.metric("Purchase Success Rate (%)", f"{success_rate}%")
        st.metric("Affordable Spending Amount", f"¥ {affordable_amount:,.2f}")
        st.info(f"Risk Assessment: {risk_level(success_rate)}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Input Overview")
        st.write(f"**Monthly Income:** ¥ {monthly_income:,.2f}")
        st.write(f"**Monthly Expense:** ¥ {monthly_expense:,.2f}")
        st.write(f"**Expense-to-Income Ratio:** {expense_ratio_pct}%")
        st.write(f"**Historical Purchase Frequency:** {purchase_frequency} times/month")
        st.write(f"**Order Cancellation Rate:** {cancel_rate_pct}%")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📌 Comparison with Database Averages")
    bcol1, bcol2, bcol3 = st.columns(3)
    with bcol1:
        st.metric("Average Monthly Spending", f"¥ {avg_expense:,.2f}")
    with bcol2:
        st.metric("Average Monthly Purchase Frequency", f"{avg_freq:.2f} times")
    with bcol3:
        st.metric("Average Cancellation Rate", f"{avg_cancel:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🛒 Recommended Product Price Range")

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""
        <div class="reco-card">
            <h4>Budget Range</h4>
            <p style="font-size:20px; font-weight:bold;">{price_reco["budget_range"]}</p>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="reco-card">
            <h4>Mid-range</h4>
            <p style="font-size:20px; font-weight:bold;">{price_reco["mid_range"]}</p>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
        <div class="reco-card">
            <h4>Premium Range</h4>
            <p style="font-size:20px; font-weight:bold;">{price_reco["premium_range"]}</p>
        </div>
        """, unsafe_allow_html=True)

    st.success(f"Recommended Positioning: {price_reco['recommended_range']}")
    st.write(f"**Advice:** {price_reco['advice']}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Advanced Visualization")

    gcol1, gcol2 = st.columns(2)

    with gcol1:
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=success_rate,
            title={'text': "Purchase Success Rate (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#4f46e5"},
                'steps': [
                    {'range': [0, 60], 'color': "#fee2e2"},
                    {'range': [60, 80], 'color': "#fef3c7"},
                    {'range': [80, 100], 'color': "#dcfce7"}
                ]
            }
        ))
        gauge_fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(gauge_fig, use_container_width=True)

    with gcol2:
        pie_fig = go.Figure(data=[go.Pie(
            labels=["Monthly Expense", "Remaining Income"],
            values=[monthly_expense, remaining_income],
            hole=0.55,
            marker=dict(colors=["#06b6d4", "#a5b4fc"])
        )])
        pie_fig.update_layout(
            title="Expense Structure",
            height=350,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(pie_fig, use_container_width=True)

    bar_fig = px.bar(
        x=["Monthly Income", "Monthly Expense", "Affordable Spending"],
        y=[monthly_income, monthly_expense, affordable_amount],
        labels={"x": "Metric", "y": "Amount"},
        color=["Monthly Income", "Monthly Expense", "Affordable Spending"],
        color_discrete_sequence=["#4f46e5", "#06b6d4", "#10b981"],
        title="Income, Expense and Affordable Spending Comparison"
    )
    bar_fig.update_layout(showlegend=False, height=420)
    st.plotly_chart(bar_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📥 Download Prediction Results")

    single_result_df = pd.DataFrame([{
        "Monthly Income": monthly_income,
        "Monthly Expense": monthly_expense,
        "Purchase Frequency": purchase_frequency,
        "Cancellation Rate (%)": cancel_rate_pct,
        "Expense-to-Income Ratio (%)": expense_ratio_pct,
        "Purchase Success Rate (%)": success_rate,
        "Affordable Spending Amount": affordable_amount,
        "Recommended Positioning": price_reco["recommended_range"],
        "Budget Range": price_reco["budget_range"],
        "Mid-range": price_reco["mid_range"],
        "Premium Range": price_reco["premium_range"],
        "Risk Level": risk_level(success_rate)
    }])

    single_csv = single_result_df.to_csv(index=False).encode("utf-8")
    st.dataframe(single_result_df, use_container_width=True)
    st.download_button(
        label="Download Single Prediction as CSV",
        data=single_csv,
        file_name="single_target_group_prediction.csv",
        mime="text/csv"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🗂 Database Preview")
    if df is not None:
        st.write("Reference only.")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.warning("high_frequency_user_profiles.csv was not found.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("👥 Multi-Group Customer Comparison")

    st.write("""
    Enter multiple customer groups below or upload a CSV file.  
    Required columns:
    - Group Name
    - Monthly Income
    - Monthly Expense
    - Purchase Frequency
    - Cancellation Rate (%)
    """)

    sample_groups = pd.DataFrame({
        "Group Name": ["Group A", "Group B", "Group C"],
        "Monthly Income": [8000, 12000, 6000],
        "Monthly Expense": [3000, 5000, 2800],
        "Purchase Frequency": [5, 8, 3],
        "Cancellation Rate (%)": [10, 6, 18]
    })

    uploaded_file = st.file_uploader("Upload group comparison CSV", type=["csv"])

    if uploaded_file is not None:
        group_df = pd.read_csv(uploaded_file)
    else:
        group_df = st.data_editor(
            sample_groups,
            num_rows="dynamic",
            use_container_width=True
        )

    required_cols = [
        "Group Name", "Monthly Income", "Monthly Expense",
        "Purchase Frequency", "Cancellation Rate (%)"
    ]

    valid_input = all(col in group_df.columns for col in required_cols)

    if valid_input and len(group_df) > 0:
        compare_df = group_df.copy()
        compare_df[[
            "Expense-to-Income Ratio (%)",
            "Purchase Success Rate (%)",
            "Affordable Spending Amount",
            "Recommended Positioning",
            "Budget Range",
            "Mid-range",
            "Premium Range",
            "Risk Level"
        ]] = compare_df.apply(process_group_row, axis=1)

        st.success("Comparison analysis generated successfully.")
        st.dataframe(compare_df, use_container_width=True)

        compare_csv = compare_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Comparison Results as CSV",
            data=compare_csv,
            file_name="multi_group_comparison_results.csv",
            mime="text/csv"
        )

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Group Comparison Charts")

        success_fig = px.bar(
            compare_df,
            x="Group Name",
            y="Purchase Success Rate (%)",
            color="Risk Level",
            title="Purchase Success Rate by Group",
            text="Purchase Success Rate (%)",
            color_discrete_map={
                "Low Risk / High Success Probability": "#10b981",
                "Medium Risk / Relatively Stable": "#f59e0b",
                "High Risk / Low Success Probability": "#ef4444"
            }
        )
        success_fig.update_traces(textposition="outside")
        success_fig.update_layout(height=420)
        st.plotly_chart(success_fig, use_container_width=True)

        amount_fig = px.bar(
            compare_df,
            x="Group Name",
            y="Affordable Spending Amount",
            color="Recommended Positioning",
            title="Affordable Spending Amount by Group",
            text="Affordable Spending Amount"
        )
        amount_fig.update_traces(texttemplate="¥%{y:.0f}", textposition="outside")
        amount_fig.update_layout(height=420)
        st.plotly_chart(amount_fig, use_container_width=True)

        scatter_fig = px.scatter(
            compare_df,
            x="Cancellation Rate (%)",
            y="Purchase Success Rate (%)",
            size="Affordable Spending Amount",
            color="Group Name",
            hover_data=["Monthly Income", "Monthly Expense", "Purchase Frequency"],
            title="Cancellation Rate vs Purchase Success Rate"
        )
        scatter_fig.update_layout(height=450)
        st.plotly_chart(scatter_fig, use_container_width=True)

        ratio_fig = px.line(
            compare_df,
            x="Group Name",
            y="Expense-to-Income Ratio (%)",
            markers=True,
            title="Expense-to-Income Ratio by Group"
        )
        ratio_fig.update_layout(height=400)
        st.plotly_chart(ratio_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        best_success_group = compare_df.loc[compare_df["Purchase Success Rate (%)"].idxmax(), "Group Name"]
        best_affordable_group = compare_df.loc[compare_df["Affordable Spending Amount"].idxmax(), "Group Name"]
        lowest_cancel_group = compare_df.loc[compare_df["Cancellation Rate (%)"].idxmin(), "Group Name"]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🏆 Key Comparison Insights")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Highest Success Rate Group", best_success_group)
        with c2:
            st.metric("Highest Affordable Spending Group", best_affordable_group)
        with c3:
            st.metric("Lowest Cancellation Rate Group", lowest_cancel_group)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.warning("Please make sure all required columns are included in the table or uploaded CSV.")
        st.markdown('</div>', unsafe_allow_html=True)