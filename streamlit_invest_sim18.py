import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timezone
from scipy.linalg import cholesky
from scipy.optimize import minimize
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO
import base64
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Define steps_per_year for monthly calculations
steps_per_year = 12

# Set wide layout and page config
st.set_page_config(page_title="Investment Sandbox Simulator (Prototype)", layout="wide", initial_sidebar_state="expanded")

# Disclaimer
st.markdown("""
    <div style="background-color: #FFCCCB; padding: 10px; border-radius: 5px;">
        <strong>Disclaimer:</strong> This is a simulation tool for educational purposes only. It is not financial advice. Consult a professional advisor before making investment decisions. Results are based on assumptions and historical data, which may not predict future performance.
    </div>
""", unsafe_allow_html=True)

# Toggle for simplified mode
simplified_mode = st.checkbox("🛡️ Use Simplified Mode (Recommended for Beginners)", value=False, help="Simplified mode hides advanced options and uses default assumptions for easier use.")

# Custom CSS for vibrant homepage with bubble animation (unchanged)
st.markdown(""" 
    <style>
    /* Hero section gradient */
    .hero {
        background: linear-gradient(135deg, #10B981, #059669);
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    .hero h1 {
        color: #F9FAFB;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .hero p {
        color: #D1D5DB;
        font-size: 1.2rem;
        margin-bottom: 1.5rem;
    }
    /* Feature cards */
    .feature-card {
        background: #1F2937;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        text-align: center;
        margin: 0.5rem;
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    .feature-card h3 {
        color: #10B981;
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .feature-card p {
        color: #D1D5DB;
    }
    /* Bubble animation */
    .bubble-container {
        position: relative;
        display: flex;
        justify-content: center;
        gap: 2rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    .bubble {
        background: #1F2937;
        color: #F9FAFB;
        border-radius: 50%;
        width: 150px;
        height: 150px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        font-size: 1rem;
        font-weight: bold;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        animation: float 4s ease-in-out infinite;
        transition: transform 0.3s ease;
    }
    .bubble:hover {
        transform: scale(1.1);
    }
    @keyframes float {
        0% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0); }
    }
    .bubble:nth-child(2) { animation-delay: 1s; }
    .bubble:nth-child(3) { animation-delay: 2s; }
    /* Button hover glow */
    .stButton > button {
        background: linear-gradient(90deg, #10B981, #059669);
        border: none;
        border-radius: 0.5rem;
        transition: box-shadow 0.3s;
    }
    .stButton > button:hover {
        box-shadow: 0 0 10px #10B981;
    }
    /* Sidebar enhancements */
    .css-1d391kg {
        background: #111827;
    }
    /* Rounded charts */
    .stPlotlyChart {
        border-radius: 0.5rem;
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# Vibrant Homepage (unchanged)
st.markdown("""
    <div class="hero">
        <h1>💸 Investment Simulator</h1>
        <p>Explore, plan, and test your financial future risk-free with interactive simulations!</p>
    </div>
""", unsafe_allow_html=True)

# Feature Cards (unchanged)
st.markdown("### Why You'll Love It")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
        <div class="feature-card">
            <h3>📈 Plan Smart</h3>
            <p>Input your savings and goals to see tailored strategies that match your vibe.</p>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
        <div class="feature-card">
            <h3>🔮 Simulate Scenarios</h3>
            <p>Run thousands of simulations to see how your portfolio might grow , or survive a crash!</p>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
        <div class="feature-card">
            <h3>💡 Learn & Grow</h3>
            <p>Get actionable advice to start investing with confidence, no real money needed.</p>
        </div>
    """, unsafe_allow_html=True)

# Bubble Section (unchanged)
st.markdown("### See Your Future")
st.markdown("""
    <div class="bubble-container">
        <div class="bubble">Plan Your Investments</div>
        <div class="bubble">Simulate Scenarios</div>
        <div class="bubble">Test Strategies Risk-Free</div>
    </div>
""", unsafe_allow_html=True)

# Helper Functions with improved error handling
def suggest_allocation(risk_profile: str, optimize=False, mus=None, cov_matrix=None, risk_free_rate=0.03):
    try:
        if optimize and mus is not None and cov_matrix is not None:
            return optimize_portfolio(risk_profile, mus, cov_matrix, risk_free_rate)
        if risk_profile == "Conservative":
            return {"Passive": 0.40, "Barbell": 0.50, "DCA": 0.10}
        if risk_profile == "Balanced":
            return {"Passive": 0.50, "Barbell": 0.30, "DCA": 0.20}
        return {"Passive": 0.30, "Barbell": 0.20, "DCA": 0.50}  # Aggressive
    except Exception as e:
        st.error(f"Error in suggesting allocation: {e}")
        return {"Passive": 0.50, "Barbell": 0.30, "DCA": 0.20}  # Default fallback

def optimize_portfolio(risk_profile, mus, cov_matrix, risk_free_rate):
    try:
        n_assets = len(mus)
        def sharpe_ratio(weights, mus, cov_matrix, risk_free_rate):
            port_return = np.sum(mus * weights)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(port_return - risk_free_rate) / port_vol if port_vol > 0 else -np.inf

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = []
        if risk_profile == "Conservative":
            bounds = [(0.3, 0.6), (0.4, 0.6), (0.0, 0.2)]
        elif risk_profile == "Balanced":
            bounds = [(0.3, 0.7), (0.2, 0.5), (0.1, 0.4)]
        else:  # Aggressive
            bounds = [(0.2, 0.5), (0.1, 0.4), (0.3, 0.7)]
        
        initial_guess = np.array([1/n_assets] * n_assets)
        result = minimize(sharpe_ratio, initial_guess, args=(mus, cov_matrix, risk_free_rate),
                          method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            return {"Passive": result.x[0], "Barbell": result.x[1], "DCA": result.x[2]}
        else:
            raise ValueError("Optimization failed")
    except Exception as e:
        st.warning(f"Optimization failed: {e}. Using default allocations.")
        return suggest_allocation(risk_profile)

def assess_risk_tolerance(answers):
    try:
        score = sum(answers)
        if score < 15:
            return "Conservative"
        elif score <= 20:
            return "Balanced"
        else:
            return "Aggressive"
    except Exception as e:
        st.error(f"Error assessing risk tolerance: {e}")
        return "Balanced"  # Default

def deterministic_growth(initials, returns, recurring, years, rebalance_freq=None, steps_per_year=12, drawdown_threshold=0.2):
    try:
        n_steps = years * steps_per_year
        values = np.zeros((len(initials), n_steps + 1))
        values[:, 0] = initials
        total_values = np.zeros(n_steps + 1)
        total_values[0] = sum(initials)
        peak_value = total_values[0]
        guardrail_triggers = 0

        target_alloc = values[:, 0] / total_values[0] if total_values[0] > 0 else np.zeros(len(initials))

        for step in range(1, n_steps + 1):
            current_value = total_values[step - 1]
            peak_value = max(peak_value, current_value)
            recur_adj = recurring if current_value >= peak_value * (1 - drawdown_threshold) else [r * 0.5 for r in recurring]
            if recur_adj != recurring:
                guardrail_triggers += 1

            for i in range(len(initials)):
                values[i, step] = values[i, step - 1] * (1 + returns[i] / steps_per_year) + recur_adj[i]
            total = values[:, step].sum()
            total_values[step] = total

            if rebalance_freq:
                rebalance_step = (12 if rebalance_freq == "Annual" else 3)
                if step % rebalance_step == 0:
                    for i in range(len(initials)):
                        values[i, step] = total * target_alloc[i]

        return total_values, guardrail_triggers
    except Exception as e:
        st.error(f"Error in deterministic growth simulation: {e}")
        return np.full(n_steps + 1, sum(initials)), 0  # Fallback flat line

def apply_progressive_tax(final_value, initial_investment, years, inflation_rate, filing_status="Single"):
    try:
        # Updated 2025 LTCG brackets
        brackets = {
            "Single": [(0, 48350, 0.0), (48350, 533400, 0.15), (533400, float('inf'), 0.20)],
            "Married": [(0, 96700, 0.0), (96700, 600050, 0.15), (600050, float('inf'), 0.20)]
        }
        adj_brackets = []
        for low, high, rate in brackets.get(filing_status, brackets["Single"]):
            adj_low = low * (1 + inflation_rate) ** years
            adj_high = high * (1 + inflation_rate) ** years if high != float('inf') else float('inf')
            adj_brackets.append((adj_low, adj_high, rate))
        
        gains = final_value - initial_investment
        if gains <= 0:
            return 0, 0.0
        tax = 0
        remaining_gains = gains
        for low, high, rate in adj_brackets:
            taxable_in_bracket = min(remaining_gains, high - low)
            if taxable_in_bracket > 0:
                tax += taxable_in_bracket * rate
                remaining_gains -= taxable_in_bracket
            if remaining_gains <= 0:
                break
        effective_tax_rate = tax / gains if gains > 0 else 0
        return tax, effective_tax_rate
    except Exception as e:
        st.error(f"Error in tax calculation: {e}")
        return 0, 0.0  # No tax fallback

def monte_carlo_rebal(initials, mus, sigmas, recurring, years, sims=1000,
                      rebalance_freq=None, steps_per_year=12, seed=None, cov_matrix=None,
                      historical_returns=None, sim_method="Normal", scenarios=None, drawdown_threshold=0.2):
    try:
        if seed is not None:
            np.random.seed(seed)
        n_assets = len(initials)
        n_steps = years * steps_per_year
        results = np.zeros((sims, n_steps + 1))
        guardrail_triggers = np.zeros(sims)
        target_alloc = np.array(initials) / sum(initials) if sum(initials) > 0 else np.zeros(n_assets)

        use_correlation = cov_matrix is not None and sim_method == "Normal"
        if use_correlation:
            L = cholesky(cov_matrix, lower=True)

        for s in range(sims):
            values = np.zeros((n_assets, n_steps + 1))
            values[:, 0] = initials
            results[s, 0] = sum(initials)
            peak_value = results[s, 0]

            for step in range(1, n_steps + 1):
                year = step / steps_per_year
                apply_scenario = scenarios and year <= 2
                if apply_scenario and "Recession" in scenarios:
                    growths = np.full(n_assets, np.exp(-0.20 / steps_per_year))
                elif apply_scenario and "Market Boom" in scenarios:
                    growths = np.full(n_assets, np.exp(0.15 / steps_per_year))
                elif sim_method == "Bootstrap" and historical_returns is not None and not historical_returns.empty:
                    sample_indices = np.random.choice(len(historical_returns), size=n_assets)
                    sampled_returns = historical_returns.iloc[sample_indices].values
                    growths = np.exp(sampled_returns)
                else:
                    if use_correlation:
                        rand_norm = np.random.standard_normal(n_assets)
                        correlated_shocks = L @ rand_norm
                        drifts = np.array(mus) / steps_per_year - 0.5 * np.array(sigmas)**2 / steps_per_year
                        diffusions = correlated_shocks * np.array(sigmas) / np.sqrt(steps_per_year)
                        growths = np.exp(drifts + diffusions)
                    else:
                        growths = np.zeros(n_assets)
                        for i in range(n_assets):
                            drift = (mus[i] - 0.5 * sigmas[i] ** 2) / steps_per_year
                            diffusion = sigmas[i] * np.sqrt(1 / steps_per_year)
                            shock = np.random.normal()
                            growths[i] = np.exp(drift + diffusion * shock)

                current_value = results[s, step - 1]
                peak_value = max(peak_value, current_value)
                recur_adj = recurring if current_value >= peak_value * (1 - drawdown_threshold) else [r * 0.5 for r in recurring]
                if recur_adj != recurring:
                    guardrail_triggers[s] += 1

                for i in range(n_assets):
                    values[i, step] = values[i, step - 1] * growths[i] + recur_adj[i]

                total = values[:, step].sum()
                results[s, step] = total

                if rebalance_freq:
                    rebalance_step = (12 if rebalance_freq == "Annual" else 3)
                    if step % rebalance_step == 0:
                        for i in range(n_assets):
                            values[i, step] = total * target_alloc[i]

        return results, guardrail_triggers
    except Exception as e:
        st.error(f"Error in Monte Carlo simulation: {e}")
        return np.full((sims, n_steps + 1), sum(initials)), np.zeros(sims)  # Fallback

def summarize_simulation(sim_matrix, guardrail_triggers=None):
    try:
        final_vals = sim_matrix[:, -1]
        summary = {
            "median": np.median(final_vals),
            "p10": np.percentile(final_vals, 10),
            "p90": np.percentile(final_vals, 90),
            "mean": np.mean(final_vals),
            "std": np.std(final_vals)
        }
        if guardrail_triggers is not None:
            summary["guardrail_trigger_pct"] = np.mean(guardrail_triggers > 0) * 100
            summary["avg_guardrail_triggers"] = np.mean(guardrail_triggers)
        return summary
    except Exception as e:
        st.error(f"Error summarizing simulation: {e}")
        return {"median": 0, "p10": 0, "p90": 0, "mean": 0, "std": 0, "guardrail_trigger_pct": 0, "avg_guardrail_triggers": 0}

def compute_efficient_frontier(mus, cov_matrix, risk_free_rate, n_points=100):
    try:
        n_assets = len(mus)
        frontier_returns = []
        frontier_vols = []
        frontier_weights = []

        def portfolio_return(weights, mus):
            return np.sum(mus * weights)
        def portfolio_volatility(weights, cov_matrix):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        min_return = min(mus)
        max_return = max(mus)
        target_returns = np.linspace(min_return, max_return, n_points)

        for target_ret in target_returns:
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: portfolio_return(w, mus) - target_ret}
            ]
            bounds = [(0, 1)] * n_assets
            initial_guess = np.array([1/n_assets] * n_assets)
            result = minimize(portfolio_volatility, initial_guess, args=(cov_matrix,),
                              method='SLSQP', bounds=bounds, constraints=constraints)
            if result.success:
                frontier_returns.append(target_ret)
                frontier_vols.append(portfolio_volatility(result.x, cov_matrix))
                frontier_weights.append(result.x)

        return frontier_returns, frontier_vols, frontier_weights
    except Exception as e:
        st.error(f"Error computing efficient frontier: {e}")
        return [], [], []

def apply_fee_drag(values, fee_rate, steps_per_year=12):
    try:
        drag_factor = (1 - fee_rate) ** (1 / steps_per_year)
        return values * drag_factor
    except Exception as e:
        st.error(f"Error applying fee drag: {e}")
        return values  # No drag fallback

def apply_inflation_adjustment(values, inflation_rate, steps_per_year=12, scenarios=None):
    try:
        adj_inflation = inflation_rate
        if scenarios and "Inflation Spike" in scenarios:
            n_steps = len(values)
            cum_inflation = np.ones(n_steps)
            for step in range(n_steps):
                year = step / steps_per_year
                rate = inflation_rate + 0.05 if year <= 2 else inflation_rate
                cum_inflation[step] = (1 + rate) ** (1 / steps_per_year)
            cum_inflation = np.cumprod(cum_inflation)
        else:
            cum_inflation = np.cumprod(np.full(len(values), (1 + inflation_rate) ** (1 / steps_per_year)))
        return values / cum_inflation
    except Exception as e:
        st.error(f"Error applying inflation adjustment: {e}")
        return values  # No adjustment fallback

def apply_tax_drag(returns, tax_rate):
    try:
        return np.array(returns) * (1 - tax_rate)
    except Exception as e:
        st.error(f"Error applying tax drag: {e}")
        return np.array(returns)

def generate_advice(risk_profile, median_real, p10_real, p90_real, cagr=None, vol=None, account_type="Taxable", scenario_impact=None, quiz_score=None, guardrail_triggers=None, years=10, lump_sum=10000, recurring=500, optimal_alloc=None, passive_alloc=0.5, barbell_alloc=0.3, dca_alloc=0.2, rebalance_freq=None):
    try:
        advice = "## Investment Advisor Guidance\n\n"
        
        advice += "### Portfolio Strategy\n"
        advice += f"Your {risk_profile.lower()} risk profile"
        if quiz_score is not None:
            advice += f" (quiz score: {quiz_score}/25)"
        advice += f" suggests a tailored approach:\n"
        advice += f"- **Expected Outcome**: Median real value after {years} years is ${median_real:,.2f} (inflation- and tax-adjusted).\n"
        advice += f"- **Risk Range**: Downside (10th percentile) at ${p10_real:,.2f}; upside (90th percentile) at ${p90_real:,.2f}.\n"
        if optimal_alloc:
            advice += f"- **Optimal Allocation**: {optimal_alloc['Passive']:.0%} Passive, {optimal_alloc['Barbell']:.0%} Barbell, {optimal_alloc['DCA']:.0%} DCA (maximizes Sharpe ratio).\n"
        if risk_profile == "Conservative":
            advice += "- **Allocation**: Increase Barbell to ~60% (safe bonds like TLT) for stability. Reduce DCA to <10% to limit volatility.\n"
            advice += "- **Action**: Prioritize low-volatility ETFs (e.g., SPY, TLT) and consider short-term bonds for capital preservation.\n"
        elif risk_profile == "Balanced":
            advice += "- **Allocation**: Maintain ~50% Passive, 30% Barbell, 20% DCA for balanced growth. Adjust if market conditions shift.\n"
            advice += "- **Action**: Diversify with broad-market ETFs (e.g., SPY) and a mix of growth (QQQ) and bonds (TLT).\n"
        else:
            advice += "- **Allocation**: Lean into DCA (~50%) and growth assets (QQQ) for higher returns, but cap Barbell at ~20%.\n"
            advice += "- **Action**: Focus on growth ETFs and consider small-cap or sector funds for higher risk-reward.\n"

        advice += "\n### Risk Management\n"
        advice += f"- **Emergency Fund**: Maintain 6 months’ expenses (~$15,000 for median U.S. household) in cash or equivalents to avoid forced sales.\n"
        if guardrail_triggers is not None:
            trigger_pct = guardrail_triggers.get("guardrail_trigger_pct", 0)
            avg_triggers = guardrail_triggers.get("avg_guardrail_triggers", 0)
            advice += f"- **Guardrails**: Triggered in {trigger_pct:.1f}% of simulations, reducing contributions by ~${recurring * 0.5 * avg_triggers:,.2f} total. Adjust threshold if too frequent.\n"
        advice += f"- **Diversification**: Spread across asset classes to reduce risk (current mix: {passive_alloc:.0%} Passive, {barbell_alloc:.0%} Barbell, {dca_alloc:.0%} DCA).\n"

        advice += "\n### Tax Considerations\n"
        advice += f"- **Account**: {account_type.lower()} account.\n"
        if account_type == "Roth":
            advice += "  - Growth and withdrawals are tax-free. Maximize contributions ($7,000/year in 2025) to save ~$3,000 in taxes over 10 years vs. taxable.\n"
        elif account_type == "Traditional":
            advice += "  - Deduct contributions now (up to $7,000/year in 2025), but plan for taxed withdrawals. Estimate future brackets for retirement.\n"
        else:
            advice += "  - Taxable accounts face capital gains taxes (15-20%). Consider tax-loss harvesting annually to offset gains (~$1,000-$2,000 savings).\n"
            advice += "  - Switch to Roth IRA for tax-free growth if eligible (income limits apply).\n"

        advice += "\n### Rebalancing and Monitoring\n"
        advice += f"- **Rebalance**: {rebalance_freq if rebalance_freq else 'No rebalancing'}. Quarterly rebalancing recommended to maintain {passive_alloc:.0%}/{barbell_alloc:.0%}/{dca_alloc:.0%} allocation.\n"
        advice += "- **Review**: Check portfolio annually or after life events (e.g., marriage, home purchase). Adjust for changing risk tolerance or goals.\n"

        if scenario_impact:
            advice += "\n### Scenario Impact\n"
            advice += f"- **{', '.join(scenario_impact['scenarios'])}**: Changes median real value by {scenario_impact['percent_change']:.1%}.\n"
            if "Recession" in scenario_impact["scenarios"]:
                advice += "- **Recession Tip**: Shift 10% from DCA to bonds (TLT) during downturns to reduce losses.\n"
            if "Inflation Spike" in scenario_impact["scenarios"]:
                advice += "- **Inflation Tip**: Consider TIPS or real assets to hedge against rising inflation.\n"
            if "Market Boom" in scenario_impact["scenarios"]:
                advice += "- **Boom Tip**: Reinvest gains but avoid chasing trends; maintain diversification.\n"

        if cagr is not None and vol is not None:
            advice += "\n### Historical Performance\n"
            advice += f"- **CAGR**: {cagr:.2%}, **Volatility**: {vol:.2%}. Use to benchmark future expectations.\n"
            advice += "- **Action**: If volatility exceeds comfort (e.g., >15%), reduce DCA or growth exposure.\n"

        advice += "\n### Next Steps\n"
        advice += "- **Consult Advisor**: Review this plan with a fiduciary advisor for personalized advice (not provided here).\n"
        advice += "- **Start Small**: Begin with ${lump_sum:,.0f} lump sum and ${recurring:,.0f}/month to build discipline.\n"
        advice += "- **Learn More**: Explore resources like Vanguard or Morningstar for investment education.\n"

        return advice
    except Exception as e:
        st.error(f"Error generating advice: {e}")
        return "Advice generation failed. Please try again."

def generate_pdf_report(inputs, metrics, advice, fig=None):
    try:
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        c.setFont("Helvetica", 12)
        y = 750
        
        c.drawString(50, y, "Investment Sandbox Simulator Report")
        c.setFont("Helvetica", 10)
        y -= 20
        c.drawString(50, y, f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        y -= 30
        
        c.drawString(50, y, "Inputs:")
        y -= 20
        for key, value in inputs.items():
            c.drawString(50, y, f"{key}: {value}")
            y -= 15
        y -= 10
        
        c.drawString(50, y, "Metrics:")
        y -= 20
        for key, value in metrics.items():
            c.drawString(50, y, f"{key}: {value}")
            y -= 15
        y -= 10
        
        c.drawString(50, y, "Advice:")
        y -= 20
        for line in advice.split('\n'):
            if y < 50:
                c.showPage()
                y = 750
            c.drawString(50, y, line)
            y -= 15
        
        if fig:
            img_buffer = BytesIO()
            fig.write_image(img_buffer, format="png")
            img_buffer.seek(0)
            img = ImageReader(img_buffer)
            if y < 300:
                c.showPage()
                y = 750
            c.drawImage(img, 50, y - 250, width=500, height=250)
        
        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Error generating PDF report: {e}")
        return None

def generate_csv_report(data, times=None, dates=None, is_backtest=False):
    try:
        if is_backtest:
            df = pd.DataFrame({
                "Date": dates,
                "Portfolio Value": data[1:]
            })
        else:
            df = pd.DataFrame(data.T, columns=[f"Sim_{i+1}" for i in range(data.shape[0])])
            df["Year"] = np.linspace(0, len(data[0])/steps_per_year, len(data[0]))
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Error generating CSV report: {e}")
        return None

@st.cache_data
def get_data(tickers, period="10y"):
    try:
        raw = yf.download(tickers, period=period, interval="1mo", progress=False, auto_adjust=False)
        if raw.empty:
            raise ValueError("No data returned")
        
        adj = pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            if 'Adj Close' in raw.columns.levels[0]:
                adj = raw['Adj Close']
            elif 'Close' in raw.columns.levels[0]:
                adj = raw['Close']
                st.warning("Using 'Close' instead of 'Adj Close'.")
            else:
                raise ValueError("No suitable price data found.")
        elif "Adj Close" in raw.columns:
            adj = raw["Adj Close"]
        elif "Close" in raw.columns:
            adj = raw["Close"]
            st.warning("Using 'Close' instead of 'Adj Close'.")
        else:
            raise ValueError("No price columns found.")
        
        adj = adj.dropna(how='all')
        return adj
    except Exception as e:
        st.error(f"Error fetching data: {e}. Trying single-ticker fallback.")
        adj = pd.DataFrame()
        for t in tickers:
            try:
                single = yf.download(t, period=period, interval="1mo", progress=False)
                if not single.empty:
                    adj[t] = single.get('Adj Close', single.get('Close'))
            except:
                pass
        if adj.empty:
            st.error("Data fetch failed completely.")
        return adj

def compute_returns(price_df):
    try:
        if price_df.empty:
            raise ValueError("Empty price DataFrame")
        monthly = price_df.resample("M").last()
        returns = np.log(monthly / monthly.shift(1)).dropna()
        return returns
    except Exception as e:
        st.error(f"Error computing returns: {e}")
        return pd.DataFrame()

def compute_historical_params(returns_df, steps_per_year=12):
    try:
        if returns_df.empty:
            raise ValueError("Empty returns DataFrame")
        ann_returns = returns_df.mean() * steps_per_year
        ann_vols = returns_df.std() * np.sqrt(steps_per_year)
        cov_ann = returns_df.cov() * steps_per_year
        corr_matrix = returns_df.corr()
        return ann_returns.values, ann_vols.values, cov_ann.values, corr_matrix
    except Exception as e:
        st.error(f"Error computing historical params: {e}")
        return None, None, None, None

def synthetic_monthly_prices(tickers, start_date, end_date, mu_map, sigma_map, seed=42):
    try:
        np.random.seed(seed)
        dates = pd.date_range(start=start_date, end=end_date, freq="M")
        out = pd.DataFrame(index=dates)
        for t in tickers:
            mu = float(mu_map.get(t, 0.07))
            sigma = float(sigma_map.get(t, 0.15))
            S0 = 100.0
            n = len(dates)
            dt = 1/12
            eps = np.random.normal(size=n)
            logret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * eps
            prices = S0 * np.exp(np.cumsum(logret))
            out[t] = prices
        return out
    except Exception as e:
        st.error(f"Error generating synthetic prices: {e}")
        return pd.DataFrame()

def backtest_portfolio(returns, allocs, initial=10000, recurring=500, drawdown_threshold=0.2):
    try:
        allocs = np.array(allocs)
        if allocs.sum() == 0:
            raise ValueError("Zero allocations")
        if returns.empty:
            raise ValueError("Empty returns")
        weights = allocs / allocs.sum()
        port_rets = (returns @ weights)

        values = [initial]
        peak_value = initial
        guardrail_triggers = 0

        for r in port_rets:
            current_value = values[-1]
            peak_value = max(peak_value, current_value)
            recur_adj = recurring if current_value >= peak_value * (1 - drawdown_threshold) else recurring * 0.5
            if recur_adj != recurring:
                guardrail_triggers += 1
            new_val = current_value * np.exp(r) + recur_adj
            values.append(new_val)
        return np.array(values), port_rets, guardrail_triggers
    except Exception as e:
        st.error(f"Error in backtest: {e}")
        return np.full(len(returns) + 1, initial), np.array([]), 0

# Tabs (unchanged structure, but with conditional rendering for simplified mode)
tab1, tab2, tab3, tab4 = st.tabs(["📊 Inputs & Risk Profile", "💼 Asset Assumptions", "🔮 Run Simulation", "📈 Backtest & Optimize"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        lump_sum = st.number_input("💰 Lump-sum amount", min_value=0.0, value=10000.0, step=100.0, format="%.2f", help="Your initial investment")
        recurring = st.number_input("📅 Recurring contribution (per month)", min_value=0.0, value=500.0, step=50.0, format="%.2f", help="Monthly additions")
        years = st.slider("⏰ Investment horizon (years)", 1, 40, 10)
    with col2:
        if simplified_mode:
            use_quiz = True  # Force quiz in simplified
            risk_profile = "Balanced"  # Default, but quiz will override
        else:
            use_quiz = st.checkbox("❓ Use risk tolerance quiz", value=True)
        if use_quiz:
            with st.expander("📝 Take the Quick Quiz", expanded=True):
                st.markdown("Answer these questions to determine your risk profile (1 = low risk, 5 = high risk):")
                q1 = st.radio("1. What is your primary investment goal?", 
                              ["Preserve capital (1)", "Balanced growth (2)", "Moderate growth (3)", "High growth (4)", "Maximize returns (5)"],
                              index=2)
                q2 = st.radio("2. How would you react to a 20% portfolio drop?", 
                              ["Very concerned, sell (1)", "Concerned, hold (2)", "Neutral (3)", "Tolerate, hold (4)", "Opportunity, buy (5)"],
                              index=2)
                q3 = st.radio("3. How stable is your income?", 
                              ["Unstable (1)", "Somewhat stable (2)", "Stable (3)", "Very stable (4)", "Extremely stable (5)"],
                              index=3)
                q4 = st.radio("4. What is your investment horizon?", 
                              ["<5 years (1)", "5-10 years (2)", "10-20 years (3)", "20+ years (4)", "30+ years (5)"],
                              index=3)
                q5 = st.radio("5. What is your investment experience?", 
                              ["None (1)", "Limited (2)", "Moderate (3)", "Experienced (4)", "Expert (5)"],
                              index=2)
                answers = [int(q.split('(')[-1][0]) for q in [q1, q2, q3, q4, q5]]
                quiz_score = sum(answers)
                risk_profile = assess_risk_tolerance(answers)
                st.success(f"🎯 Quiz Score: {quiz_score}/25 → **{risk_profile}** Profile")
        else:
            risk_profile = st.selectbox("🎯 Risk profile", ["Balanced", "Conservative", "Aggressive"])
            quiz_score = None

        if not simplified_mode:
            use_optimization = st.checkbox("⚡ Use portfolio optimization", value=False)
            risk_free_rate = st.number_input("🏦 Risk-free rate (%)", min_value=0.0, value=3.0, step=0.1, format="%.2f") / 100
        else:
            use_optimization = False
            risk_free_rate = 0.03
            quiz_score = quiz_score  # From quiz

    with st.expander("⚖️ Suggested Allocation (Editable)", expanded=not simplified_mode):
        preset_alloc = suggest_allocation(risk_profile)
        if simplified_mode:
            passive_pct = int(preset_alloc["Passive"] * 100)
            barbell_pct = int(preset_alloc["Barbell"] * 100)
            dca_pct = int(preset_alloc["DCA"] * 100)
        else:
            col3, col4, col5 = st.columns(3)
            with col3:
                passive_pct = st.slider("📉 Passive ETF %", 0, 100, int(preset_alloc["Passive"] * 100), help="Broad market exposure")
            with col4:
                barbell_pct = st.slider("⚖️ Barbell %", 0, 100, int(preset_alloc["Barbell"] * 100), help="Safe + growth mix")
            with col5:
                dca_pct = st.slider("💧 DCA-focused %", 0, 100, int(preset_alloc["DCA"] * 100), help="Dollar-cost averaging")

        total = passive_pct + barbell_pct + dca_pct
        if total == 0:
            passive_pct, barbell_pct, dca_pct = 50, 30, 20
            total = 100
        passive_alloc = passive_pct / total
        barbell_alloc = barbell_pct / total
        dca_alloc = dca_pct / total
        st.info(f"✅ Normalized: **Passive: {passive_alloc:.0%}**, **Barbell: {barbell_alloc:.0%}**, **DCA: {dca_alloc:.0%}**")

    if not simplified_mode:
        with st.expander("⚙️ Fees & Adjustments"):
            col6, col7 = st.columns(2)
            with col6:
                fee_profile = st.selectbox("💸 Fee level", ["Typical", "Low", "High"])
                if fee_profile == "Low":
                    expense_ratio = 0.003
                elif fee_profile == "High":
                    expense_ratio = 0.01
                else:
                    expense_ratio = 0.006
                inflation_rate = st.number_input("📈 Annual inflation rate", min_value=0.0, value=0.02, step=0.005, format="%.3f")
            with col7:
                drawdown_threshold = st.number_input("🛡️ Guardrail drawdown threshold (%)", min_value=0.0, max_value=50.0, value=20.0, step=5.0, format="%.1f") / 100
                filing_status = st.selectbox("📋 Filing status", ["Single", "Married"], help="For tax bracket calculations")
                account_type = st.selectbox("🏦 Account type", ["Taxable", "Traditional", "Roth"], help="Taxable: annual taxes; Traditional: tax-deferred; Roth: tax-free withdrawals")
                tax_rate = st.number_input("💰 Fallback tax rate (if needed)", min_value=0.0, value=0.15, step=0.01, format="%.2f")
    else:
        expense_ratio = 0.006  # Typical
        inflation_rate = 0.02
        drawdown_threshold = 0.20
        filing_status = "Single"
        account_type = "Taxable"
        tax_rate = 0.15

    if not simplified_mode:
        with st.expander("🎲 Simulation Settings"):
            col8, col9 = st.columns(2)
            with col8:
                model = st.selectbox("📊 Model", ["Deterministic", "Monte Carlo"])
                sims = st.number_input("🔄 Monte Carlo simulations", min_value=500, max_value=20000, value=2000, step=500)
                sim_method = st.selectbox("🔬 Monte Carlo method", ["Normal (GBM)", "Bootstrap"], help="Bootstrap uses historical returns for realism; Normal assumes standard distribution.")
            with col9:
                rebalance_freq = st.selectbox("🔄 Rebalancing frequency", [None, "Annual", "Quarterly"])
                scenarios = st.multiselect("🌪️ Stress scenarios (first 2 years)", ["Recession", "Inflation Spike", "Market Boom"], help="Recession: -20% returns; Inflation Spike: +5% inflation; Market Boom: +15% returns")
                use_historical = st.checkbox("📚 Use historical params for accuracy (requires data fetch)", value=True)
    else:
        model = "Monte Carlo"
        sims = 1000
        sim_method = "Normal (GBM)"
        rebalance_freq = "Annual"
        scenarios = []
        use_historical = True

    if not simplified_mode and st.button("🔍 Run Data Diagnostics"):
        with st.spinner("Checking connections..."):
            try:
                r = requests.get("https://query1.finance.yahoo.com", timeout=5)
                st.success(f"✅ HTTP to Yahoo Finance OK (status: {r.status_code})")
            except Exception as e:
                st.error(f"❌ HTTP test failed: {e}")

            try:
                test = yf.download(["SPY","TLT","QQQ"], period="1mo", interval="1mo", progress=False, auto_adjust=False)
                st.success(f"✅ yf.download shape: {test.shape if test is not None else 'None'}")
                if isinstance(test, pd.DataFrame) and not test.empty:
                    st.write("📋 Preview (top):")
                    st.dataframe(test.head().fillna("NaN"))
            except Exception as e:
                st.error(f"❌ yfinance download failed: {e}")

with tab2:
    if simplified_mode:
        passive_ticker = "SPY"
        passive_return = 0.07
        passive_vol = 0.15
        bond_ticker = "TLT"
        bond_return = 0.03
        bond_vol = 0.04
        growth_ticker = "QQQ"
        growth_return = 0.09
        growth_vol = 0.20
        dca_ticker = "QQQ"
        dca_return = 0.10
        dca_vol = 0.22
        st.info("In simplified mode, default asset assumptions are used.")
    else:
        col10, col11, col12 = st.columns(3)
        with col10:
            st.subheader("📉 Passive ETF")
            passive_ticker = st.text_input("Ticker", value="SPY", help="e.g., SPY for S&P 500")
            passive_return = st.number_input("Annual return", value=0.07, step=0.005, format="%.3f")
            passive_vol = st.number_input("Volatility", value=0.15, step=0.01, format="%.2f")
        with col11:
            st.subheader("⚖️ Barbell")
            bond_ticker = st.text_input("Safe ticker", value="TLT", help="e.g., TLT for bonds")
            bond_return = st.number_input("Bond annual return", value=0.03, step=0.005, format="%.3f")
            bond_vol = st.number_input("Bond volatility", value=0.04, step=0.01, format="%.2f")
            growth_ticker = st.text_input("Growth ticker", value="QQQ", help="e.g., QQQ for tech")
            growth_return = st.number_input("Growth annual return", value=0.09, step=0.005, format="%.3f")
            growth_vol = st.number_input("Growth volatility", value=0.20, step=0.01, format="%.2f")
        with col12:
            st.subheader("💧 DCA-Focused")
            dca_ticker = st.text_input("Ticker", value="QQQ")
            dca_return = st.number_input("Annual return", value=0.10, step=0.005, format="%.3f")
            dca_vol = st.number_input("Volatility", value=0.22, step=0.01, format="%.2f")

    tickers = [passive_ticker, bond_ticker, growth_ticker, dca_ticker]

    initial_passive = lump_sum * passive_alloc
    initial_barbell = lump_sum * barbell_alloc
    initial_dca = lump_sum * dca_alloc
    barbell_safe = initial_barbell * 0.6
    barbell_growth = initial_barbell * 0.4

    initials = [initial_passive, barbell_safe, barbell_growth, initial_dca]
    recurs = [recurring * passive_alloc, recurring * barbell_alloc * 0.6, recurring * barbell_alloc * 0.4, recurring * dca_alloc]

    st.subheader("💼 Allocation Summary")
    col13, col14, col15, col16 = st.columns(4)
    col13.metric("Passive", f"${initial_passive:,.2f}", help=passive_ticker)
    col14.metric("Barbell Safe", f"${barbell_safe:,.2f}", help=bond_ticker)
    col15.metric("Barbell Growth", f"${barbell_growth:,.2f}", help=growth_ticker)
    col16.metric("DCA", f"${initial_dca:,.2f}", help=dca_ticker)

with tab3:
    optimal_alloc = None
    if not simplified_mode and st.button("⚡ Run Portfolio Optimization", use_container_width=True):
        with st.spinner("Optimizing..."):
            mus = [passive_return, bond_return, growth_return, dca_return]
            sigmas = [passive_vol, bond_vol, growth_vol, dca_vol]
            cov_matrix = np.diag(np.array(sigmas) ** 2)  # Default
            if use_historical:
                unique_tickers = list(set(tickers))
                if unique_tickers:
                    prices = get_data(unique_tickers, period="max")
                    if not prices.empty:
                        returns = compute_returns(prices)
                        if not returns.empty:
                            returns_3 = pd.DataFrame({
                                'passive': returns[passive_ticker],
                                'barbell': 0.6 * returns[bond_ticker] + 0.4 * returns[growth_ticker],
                                'dca': returns[dca_ticker]
                            }).dropna()
                            hist_mus, hist_sigmas, cov_matrix, corr_df = compute_historical_params(returns_3, steps_per_year)
                            if hist_mus is not None:
                                mus = hist_mus
                                sigmas = hist_sigmas
                                st.success("✅ Using historical params for optimization!")
                                st.subheader("🔗 Historical Correlation Matrix")
                                st.dataframe(corr_df.round(3))
            
            optimal_alloc = optimize_portfolio(risk_profile, mus, cov_matrix, risk_free_rate)
            st.subheader("🎯 Optimized Allocation")
            st.info(f"**Optimal: {optimal_alloc['Passive']:.0%} Passive, {optimal_alloc['Barbell']:.0%} Barbell, {optimal_alloc['DCA']:.0%} DCA**")
            opt_return = np.sum(np.array(mus) * np.array([optimal_alloc['Passive'], optimal_alloc['Barbell'], optimal_alloc['DCA']]))
            opt_vol = np.sqrt(np.dot([optimal_alloc['Passive'], optimal_alloc['Barbell'], optimal_alloc['DCA']], 
                                     np.dot(cov_matrix, [optimal_alloc['Passive'], optimal_alloc['Barbell'], optimal_alloc['DCA']])))
            opt_sharpe = (opt_return - risk_free_rate) / opt_vol if opt_vol > 0 else 0

            col17, col18, col19 = st.columns(3)
            col17.metric("📈 Expected Return", f"{opt_return:.2%}")
            col18.metric("📊 Volatility", f"{opt_vol:.2%}")
            col19.metric("⚡ Sharpe Ratio", f"{opt_sharpe:.2f}")

            frontier_returns, frontier_vols, _ = compute_efficient_frontier(mus, cov_matrix, risk_free_rate)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=frontier_vols, y=frontier_returns, mode='lines', name='Efficient Frontier', line=dict(color='#10B981')))
            fig.add_trace(go.Scatter(x=[opt_vol], y=[opt_return], mode='markers', name='Optimal Portfolio', marker=dict(size=12, color='#10B981')))
            fig.add_trace(go.Scatter(x=[np.sqrt(np.dot([passive_alloc, barbell_alloc, dca_alloc], 
                                                       np.dot(cov_matrix, [passive_alloc, barbell_alloc, dca_alloc])))],
                                     y=[np.sum(np.array(mus) * np.array([passive_alloc, barbell_alloc, dca_alloc]))],
                                     mode='markers', name='Current Portfolio', marker=dict(size=12, symbol='x', color='gray')))
            fig.update_layout(title="📈 Efficient Frontier", xaxis_title="Volatility", yaxis_title="Expected Return", hovermode='closest', showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

    if st.button("🎲 Run Simulation", use_container_width=True):
        cov_matrix = None
        mus = [passive_return, bond_return, growth_return, dca_return]
        sigmas = [passive_vol, bond_vol, growth_vol, dca_vol]
        historical_returns = None
        total_initial = sum(initials)

        mus = apply_tax_drag(mus, tax_rate) if account_type == "Taxable" else mus

        if use_historical:
            unique_tickers = list(set(tickers))
            if unique_tickers:
                period_arg = "max"
                prices = get_data(unique_tickers, period=period_arg)
                if not prices.empty:
                    returns = compute_returns(prices)
                    if not returns.empty:
                        returns_4 = pd.DataFrame({
                            'passive': returns[passive_ticker],
                            'bond': returns[bond_ticker],
                            'growth': returns[growth_ticker],
                            'dca': returns[dca_ticker]
                        }).dropna()
                        hist_mus, hist_sigmas, cov_matrix, corr_df = compute_historical_params(returns_4, steps_per_year)
                        if hist_mus is not None:
                            mus = apply_tax_drag(hist_mus, tax_rate) if account_type == "Taxable" else hist_mus
                            sigmas = hist_sigmas
                            historical_returns = returns_4
                            if not simplified_mode:
                                st.success("✅ Using historical params for accuracy!")
                                st.subheader("🔗 Historical Correlation Matrix")
                                st.dataframe(corr_df.round(3))
                else:
                    st.warning("⚠️ Historical data unavailable; using user inputs.")

        if use_optimization and cov_matrix is not None:
            optimal_alloc = optimize_portfolio(risk_profile, mus[:3], cov_matrix[:3, :3], risk_free_rate)  # Adjust for 3 assets
            passive_alloc = optimal_alloc["Passive"]
            barbell_alloc = optimal_alloc["Barbell"]
            dca_alloc = optimal_alloc["DCA"]
            initials = [lump_sum * passive_alloc, lump_sum * barbell_alloc * 0.6, lump_sum * barbell_alloc * 0.4, lump_sum * dca_alloc]
            recurs = [recurring * passive_alloc, recurring * barbell_alloc * 0.6, recurring * barbell_alloc * 0.4, recurring * dca_alloc]
            if not simplified_mode:
                st.info(f"⚡ Using optimized allocation: {passive_alloc:.0%} Passive, {barbell_alloc:.0%} Barbell, {dca_alloc:.0%} DCA")

        if sim_method == "Bootstrap" and historical_returns is None:
            st.warning("⚠️ Bootstrap selected but no historical data available; falling back to Normal (GBM).")
            sim_method = "Normal"

        if model == "Deterministic":
            trajectory, guardrail_triggers = deterministic_growth(initials, mus, recurs, years,
                                                                  rebalance_freq=rebalance_freq, steps_per_year=steps_per_year,
                                                                  drawdown_threshold=drawdown_threshold)
            trajectory_with_fees = apply_fee_drag(trajectory, expense_ratio, steps_per_year)
            trajectory_real = apply_inflation_adjustment(trajectory_with_fees, inflation_rate, steps_per_year, scenarios)
            tax_amount, effective_tax_rate = apply_progressive_tax(trajectory_real[-1], total_initial, years, inflation_rate, filing_status) if account_type != "Roth" else (0, 0)
            trajectory_post_tax = trajectory_real[-1] - tax_amount if account_type != "Roth" else trajectory_real[-1]
            fee_cost = trajectory[-1] - trajectory_with_fees[-1] if len(trajectory) > 0 else 0

            st.subheader("📊 Deterministic Projection")
            col20, col21, col22, col23, col24 = st.columns(5)
            col20.metric("💵 Projected Nominal (net fees)", f"${trajectory_with_fees[-1]:,.2f}" if len(trajectory_with_fees) > 0 else "$0")
            col21.metric("🌟 Projected Real (net inflation/taxes)", f"${trajectory_post_tax:,.2f}")
            col22.metric("💸 Total Fees Paid", f"${fee_cost:,.2f}")
            col23.metric("🧾 Effective Tax Rate", f"{effective_tax_rate:.2%}")
            col24.metric("🛡️ Guardrail Triggers", f"{guardrail_triggers} times")

            times = np.linspace(0, years, len(trajectory))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=times, y=trajectory, mode='lines', name="Before fees", line=dict(dash='dash', color='gray')))
            fig.add_trace(go.Scatter(x=times, y=trajectory_with_fees, mode='lines', name="After fees", line=dict(color='#10B981')))
            fig.add_trace(go.Scatter(x=times, y=trajectory_real, mode='lines', name="After inflation/taxes", line=dict(color='#059669')))
            fig.update_layout(title="📈 Deterministic Portfolio Projection", xaxis_title="Years", yaxis_title="Portfolio Value ($)", hovermode='x unified', showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("💡 Advice")
            advice = generate_advice(risk_profile, trajectory_post_tax, trajectory_post_tax * 0.8, trajectory_post_tax * 1.2,
                                     account_type=account_type, quiz_score=quiz_score, guardrail_triggers={"guardrail_trigger_pct": guardrail_triggers/len(trajectory)*100 if len(trajectory) > 0 else 0, "avg_guardrail_triggers": guardrail_triggers},
                                     years=years, lump_sum=lump_sum, recurring=recurring, optimal_alloc=optimal_alloc, passive_alloc=passive_alloc, barbell_alloc=barbell_alloc, dca_alloc=dca_alloc, rebalance_freq=rebalance_freq)
            st.markdown(advice)

            inputs_dict = {
                "Lump Sum": f"${lump_sum:,.2f}",
                "Recurring (Monthly)": f"${recurring:,.2f}",
                "Years": years,
                "Risk Profile": risk_profile,
                "Account Type": account_type,
                "Scenarios": ", ".join(scenarios) if scenarios else "None",
                "Drawdown Threshold": f"{drawdown_threshold*100:.1f}%",
                "Optimized Allocation": f"{optimal_alloc['Passive']:.0%} Passive, {optimal_alloc['Barbell']:.0%} Barbell, {optimal_alloc['DCA']:.0%} DCA" if optimal_alloc else "Not used"
            }
            metrics_dict = {
                "Nominal Value (net fees)": f"${trajectory_with_fees[-1]:,.2f}" if len(trajectory_with_fees) > 0 else "$0",
                "Real Value (net inflation/taxes)": f"${trajectory_post_tax:,.2f}",
                "Fees Paid": f"${fee_cost:,.2f}",
                "Effective Tax Rate": f"{effective_tax_rate:.2%}",
                "Guardrail Triggers": f"{guardrail_triggers} times"
            }
            pdf_buffer = generate_pdf_report(inputs_dict, metrics_dict, advice, fig)
            if pdf_buffer:
                st.download_button("📥 Download PDF Report", data=pdf_buffer, file_name="investment_report.pdf", mime="application/pdf")
            csv_buffer = generate_csv_report(np.array([trajectory]), times=times)
            if csv_buffer:
                st.download_button("📊 Download CSV Data", data=csv_buffer, file_name="investment_data.csv", mime="text/csv")

        else:
            sim_results, guardrail_triggers = monte_carlo_rebal(initials, mus, sigmas, recurs, years,
                                                                sims=sims, rebalance_freq=rebalance_freq,
                                                                steps_per_year=steps_per_year,
                                                                cov_matrix=cov_matrix,
                                                                historical_returns=historical_returns, sim_method=sim_method,
                                                                scenarios=scenarios, drawdown_threshold=drawdown_threshold)
            sim_results_with_fees = apply_fee_drag(sim_results, expense_ratio, steps_per_year)
            sim_results_real = apply_inflation_adjustment(sim_results_with_fees, inflation_rate, steps_per_year, scenarios)
            summary = summarize_simulation(sim_results_with_fees, guardrail_triggers)
            summary_real = summarize_simulation(sim_results_real)
            tax_amount, effective_tax_rate = apply_progressive_tax(summary_real["median"], total_initial, years, inflation_rate, filing_status) if account_type != "Roth" else (0, 0)
            median_post_tax = summary_real["median"] - tax_amount if account_type != "Roth" else summary_real["median"]
            fee_cost = np.median(sim_results[:, -1]) - summary["median"] if sim_results.size > 0 else 0

            scenario_summary = None
            if scenarios:
                sim_results_scen, scen_guardrail_triggers = monte_carlo_rebal(initials, mus, sigmas, recurs, years,
                                                                              sims=sims, rebalance_freq=rebalance_freq,
                                                                              steps_per_year=steps_per_year, cov_matrix=cov_matrix,
                                                                              historical_returns=historical_returns, sim_method=sim_method,
                                                                              scenarios=scenarios, drawdown_threshold=drawdown_threshold)
                sim_results_scen_fees = apply_fee_drag(sim_results_scen, expense_ratio, steps_per_year)
                sim_results_scen_real = apply_inflation_adjustment(sim_results_scen_fees, inflation_rate, steps_per_year, scenarios)
                scenario_summary = summarize_simulation(sim_results_scen_real, scen_guardrail_triggers)
                scen_tax, _ = apply_progressive_tax(scenario_summary["median"], total_initial, years, inflation_rate, filing_status) if account_type != "Roth" else (0, 0)
                scen_median_post_tax = scenario_summary["median"] - scen_tax if account_type != "Roth" else scenario_summary["median"]

            st.subheader(f"Monte Carlo Projection ({sim_method}; with correlations if historical data used)")
            col25, col26, col27, col28, col29, col30 = st.columns(6)
            col25.metric("Median nominal (net of fees)", f"${summary['median']:,.2f}")
            col26.metric("Median real (net of inflation/taxes)", f"${median_post_tax:,.2f}")
            col27.metric("10th percentile real", f"${summary_real['p10']:,.2f}")
            col28.metric("90th percentile real", f"${summary_real['p90']:,.2f}")
            col29.metric("Median fees paid", f"${fee_cost:,.2f}")
            col30.metric("Effective tax rate", f"{effective_tax_rate:.2%}")
            st.metric("🛡️ Guardrail Triggers", f"{summary['guardrail_trigger_pct']:.1f}% of sims (avg {summary['avg_guardrail_triggers']:.1f} times)")

            times = np.linspace(0, years, sim_results.shape[1])
            fig = go.Figure()
            percentiles = np.percentile(sim_results_with_fees, [10, 50, 90], axis=0)
            percentiles_real = np.percentile(sim_results_real, [10, 50, 90], axis=0)
            fig.add_trace(go.Scatter(x=times, y=percentiles[1], mode='lines', name="Median nominal (net fees)", line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=times, y=percentiles_real[1], mode='lines', name="Median real (net inflation/taxes)"))
            fig.add_trace(go.Scatter(x=times, y=percentiles_real[2], mode='none', name="90th real", fill='tonexty', fillcolor='rgba(0,100,80,0.2)'))
            fig.add_trace(go.Scatter(x=times, y=percentiles_real[0], mode='none', name="10th real", fill='tonexty', fillcolor='rgba(100,0,80,0.2)'))
            
            if scenario_summary:
                percentiles_scen = np.percentile(sim_results_scen_real, [10, 50, 90], axis=0)
                fig.add_trace(go.Scatter(x=times, y=percentiles_scen[1], mode='lines', name=f"Median real ({', '.join(scenarios)})", line=dict(dash='dot')))
                fig.add_trace(go.Scatter(x=times, y=percentiles_scen[2], mode='none', name=f"90th real ({', '.join(scenarios)})", fill='tonexty', fillcolor='rgba(255,165,0,0.1)'))
                fig.add_trace(go.Scatter(x=times, y=percentiles_scen[0], mode='none', name=f"10th real ({', '.join(scenarios)})", fill='tonexty', fillcolor='rgba(255,99,71,0.1)'))

            fig.update_layout(title="📈 Monte Carlo Portfolio Projection", xaxis_title="Years", yaxis_title="Portfolio Value ($)", hovermode='x unified', showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("💡 Advice")
            scenario_impact = None
            if scenario_summary:
                percent_change = (scenario_summary["median"] - summary_real["median"]) / summary_real["median"] if summary_real["median"] > 0 else 0
                scenario_impact = {"scenarios": scenarios, "percent_change": percent_change}
                st.metric("🌪️ Scenario Impact", f"{percent_change:.1%} change in median real value")
            advice = generate_advice(risk_profile, median_post_tax, summary_real['p10'], summary_real['p90'],
                                     account_type=account_type, scenario_impact=scenario_impact, quiz_score=quiz_score,
                                     guardrail_triggers=summary, years=years, lump_sum=lump_sum, recurring=recurring, optimal_alloc=optimal_alloc,
                                     passive_alloc=passive_alloc, barbell_alloc=barbell_alloc, dca_alloc=dca_alloc, rebalance_freq=rebalance_freq)
            st.markdown(advice)

            inputs_dict = {
                "Lump Sum": f"${lump_sum:,.2f}",
                "Recurring (Monthly)": f"${recurring:,.2f}",
                "Years": years,
                "Risk Profile": risk_profile,
                "Account Type": account_type,
                "Scenarios": ", ".join(scenarios) if scenarios else "None",
                "Simulation Method": sim_method,
                "Drawdown Threshold": f"{drawdown_threshold*100:.1f}%",
                "Optimized Allocation": f"{optimal_alloc['Passive']:.0%} Passive, {optimal_alloc['Barbell']:.0%} Barbell, {optimal_alloc['DCA']:.0%} DCA" if optimal_alloc else "Not used"
            }
            metrics_dict = {
                "Median Nominal (net fees)": f"${summary['median']:,.2f}",
                "Median Real (net inflation/taxes)": f"${median_post_tax:,.2f}",
                "10th Percentile Real": f"${summary_real['p10']:,.2f}",
                "90th Percentile Real": f"${summary_real['p90']:,.2f}",
                "Fees Paid": f"${fee_cost:,.2f}",
                "Effective Tax Rate": f"{effective_tax_rate:.2%}",
                "Guardrail Triggers": f"{summary['guardrail_trigger_pct']:.1f}% of sims (avg {summary['avg_guardrail_triggers']:.1f} times)"
            }
            if scenario_impact:
                metrics_dict["Scenario Impact"] = f"{scenario_impact['percent_change']:.1%} ({', '.join(scenarios)})"
            pdf_buffer = generate_pdf_report(inputs_dict, metrics_dict, advice, fig)
            if pdf_buffer:
                st.download_button("📥 Download PDF Report", data=pdf_buffer, file_name="investment_report.pdf", mime="application/pdf")

            csv_buffer = generate_csv_report(sim_results, times=times)
            if csv_buffer:
                st.download_button("📊 Download CSV Data", data=csv_buffer, file_name="investment_data.csv", mime="text/csv")

with tab4:
    if simplified_mode:
        backtest_period = "10y"
        allow_synthetic = True
    else:
        col31, col32 = st.columns(2)
        with col31:
            backtest_period = st.selectbox("📅 Backtest Period", ["5y", "10y", "Max"], index=1)
        with col32:
            allow_synthetic = st.checkbox("🔄 Allow synthetic fallback if real data fails", value=True)

    if st.button("📈 Run Backtest with Real Data", use_container_width=True):
        with st.spinner("Fetching historical data..."):
            st.subheader(f"📈 Historical Backtest ({passive_ticker}, {bond_ticker}, {growth_ticker}, {dca_ticker})")

            unique_tickers = list(set(tickers))
            period_map = {"5y": "5y", "10y": "10y", "Max": "max"}
            period_arg = period_map.get(backtest_period, "10y")

            prices = get_data(unique_tickers, period=period_arg)

            if not simplified_mode:
                st.markdown("---")
                st.markdown("**📋 Price data preview (first 5 rows)**")
                if not prices.empty:
                    st.dataframe(prices.head())
                    returns = compute_returns(prices)
                    if not returns.empty:
                        returns_4 = pd.DataFrame({
                            'passive': returns[passive_ticker],
                            'bond': returns[bond_ticker],
                            'growth': returns[growth_ticker],
                            'dca': returns[dca_ticker]
                        }).dropna()
                        _, _, _, corr_df = compute_historical_params(returns_4, steps_per_year)
                        st.subheader("🔗 Historical Correlation Matrix (for accuracy)")
                        st.dataframe(corr_df.round(3))

            if prices.empty and allow_synthetic:
                st.warning("⚠️ No real price data available , using synthetic demo data (so backtest can run).")
                end = datetime.today()
                if period_arg == "5y":
                    start = end - pd.DateOffset(years=5)
                elif period_arg == "10y":
                    start = end - pd.DateOffset(years=10)
                else:
                    start = end - pd.DateOffset(years=15)
                mu_map = {passive_ticker: passive_return, bond_ticker: bond_return, growth_ticker: growth_return, dca_ticker: dca_return}
                sigma_map = {passive_ticker: passive_vol, bond_ticker: bond_vol, growth_ticker: growth_vol, dca_ticker: dca_vol}
                prices = synthetic_monthly_prices(unique_tickers, start, end, mu_map, sigma_map)
                st.info("🔄 Synthetic data generated. (This is not real market data.)")
                if not simplified_mode:
                    st.dataframe(prices.head())

            if prices.empty:
                st.warning("❌ Cannot run backtest because no price data is available. Try the diagnostics button in the sidebar.")
            else:
                returns = compute_returns(prices)
                if not simplified_mode:
                    st.markdown("---")
                    st.markdown("**📈 Returns preview (first 5 rows)**")
                    if not returns.empty:
                        st.dataframe(returns.head())
                    else:
                        st.warning("⚠️ Price data exists but returns could not be computed (not enough history).")

                if not returns.empty:
                    effective_returns = pd.DataFrame({
                        'passive': returns[passive_ticker],
                        'barbell': 0.6 * returns[bond_ticker] + 0.4 * returns[growth_ticker],
                        'dca': returns[dca_ticker]
                    }).dropna()
                    allocs = [passive_alloc, barbell_alloc, dca_alloc]
                    if use_optimization:
                        hist_mus, hist_sigmas, cov_matrix, _ = compute_historical_params(effective_returns, steps_per_year)
                        if cov_matrix is not None:
                            optimal_alloc = optimize_portfolio(risk_profile, hist_mus, cov_matrix, risk_free_rate)
                            allocs = [optimal_alloc['Passive'], optimal_alloc['Barbell'], optimal_alloc['DCA']]
                            if not simplified_mode:
                                st.info(f"⚡ Using optimized allocation: {allocs[0]:.0%} Passive, {allocs[1]:.0%} Barbell, {allocs[2]:.0%} DCA")
                    values, port_rets, guardrail_triggers = backtest_portfolio(
                        effective_returns,
                        allocs=allocs,
                        initial=lump_sum,
                        recurring=recurring,
                        drawdown_threshold=drawdown_threshold
                    )

                    if len(port_rets) > 0:
                        values_with_fees = apply_fee_drag(values, expense_ratio, steps_per_year)
                        values_real = apply_inflation_adjustment(values_with_fees, inflation_rate, steps_per_year, scenarios)
                        tax_amount, effective_tax_rate = apply_progressive_tax(values_real[-1], lump_sum, years, inflation_rate, filing_status) if account_type != "Roth" else (0, 0)
                        values_post_tax = values_real[-1] - tax_amount if account_type != "Roth" else values_real[-1]

                        dates = effective_returns.index
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=dates, y=values[1:], mode='lines', name="Backtest Portfolio", line=dict(color='#10B981')))
                        fig.update_layout(title="📈 Historical Backtest Portfolio Growth", xaxis_title="Date", yaxis_title="Portfolio Value ($)", hovermode='x unified', showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)

                        years_count = len(port_rets) / 12
                        if years_count > 0:
                            cagr = (values[-1] / lump_sum) ** (1 / years_count) - 1 if lump_sum > 0 else 0
                            vol = np.std(port_rets) * np.sqrt(12) if len(port_rets) > 0 else 0
                            col33, col34, col35, col36 = st.columns(4)
                            col33.metric("📈 CAGR", f"{cagr:.2%}")
                            col34.metric("📊 Volatility", f"{vol:.2%}")
                            col35.metric("🌟 Final real value (net of inflation/taxes)", f"${values_post_tax:,.2f}")
                            col36.metric("🧾 Effective tax rate", f"{effective_tax_rate:.2%}")
                            st.metric("🛡️ Guardrail Triggers", f"{guardrail_triggers} times")
                            st.subheader("💡 Advice from Backtest")
                            advice = generate_advice(risk_profile, values_post_tax, values_post_tax * 0.8, values_post_tax * 1.2,
                                                     cagr, vol, account_type=account_type, quiz_score=quiz_score,
                                                     guardrail_triggers={"guardrail_trigger_pct": guardrail_triggers/len(values)*100 if len(values) > 0 else 0, "avg_guardrail_triggers": guardrail_triggers},
                                                     years=years, lump_sum=lump_sum, recurring=recurring, optimal_alloc=optimal_alloc,
                                                     passive_alloc=passive_alloc, barbell_alloc=barbell_alloc, dca_alloc=dca_alloc, rebalance_freq=rebalance_freq)
                            st.markdown(advice)

                            inputs_dict = {
                                "Lump Sum": f"${lump_sum:,.2f}",
                                "Recurring (Monthly)": f"${recurring:,.2f}",
                                "Period": backtest_period,
                                "Risk Profile": risk_profile,
                                "Account Type": account_type,
                                "Drawdown Threshold": f"{drawdown_threshold*100:.1f}%",
                                "Optimized Allocation": f"{optimal_alloc['Passive']:.0%} Passive, {optimal_alloc['Barbell']:.0%} Barbell, {optimal_alloc['DCA']:.0%} DCA" if optimal_alloc else "Not used"
                            }
                            metrics_dict = {
                                "Final Value (net fees)": f"${values_with_fees[-1]:,.2f}" if len(values_with_fees) > 0 else "$0",
                                "Final Real Value (net inflation/taxes)": f"${values_post_tax:,.2f}",
                                "CAGR": f"{cagr:.2%}",
                                "Volatility": f"{vol:.2%}",
                                "Effective Tax Rate": f"{effective_tax_rate:.2%}",
                                "Guardrail Triggers": f"{guardrail_triggers} times"
                            }
                            pdf_buffer = generate_pdf_report(inputs_dict, metrics_dict, advice, fig)
                            if pdf_buffer:
                                st.download_button("📥 Download PDF Report", data=pdf_buffer, file_name="backtest_report.pdf", mime="application/pdf")

                            csv_buffer = generate_csv_report(values, dates=dates, is_backtest=True)
                            if csv_buffer:
                                st.download_button("📊 Download CSV Data", data=csv_buffer, file_name="backtest_data.csv", mime="text/csv")
                        else:
                            st.warning("⚠️ Not enough return periods to compute CAGR/Volatility.")
                    else:
                        st.warning("⚠️ Not enough data to compute backtest results. Try a longer period or check your diagnostics.")
                else:
                    st.warning("⚠️ No returns data available for backtest.")


st.caption(f"✨ Prototype upgraded 2025-09-20 21:20 UTC")
