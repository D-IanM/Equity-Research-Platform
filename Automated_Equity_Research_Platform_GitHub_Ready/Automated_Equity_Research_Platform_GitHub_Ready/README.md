# Automated Equity Research Platform

The **Automated Equity Research Platform** is a GitHub-ready Python project that analyzes publicly traded companies through a complete equity research workflow. It collects market and financial data, calculates financial ratios, performs DCF valuation, compares companies with peers, measures investment risk, optimizes portfolios, creates interactive dashboards, and generates automated Markdown/HTML research reports.

The project now supports two ways to use it:

1. **Streamlit web app** — users open a normal website link and run the dashboard directly.
2. **Standalone Google Colab notebooks** — each notebook can run by itself for a specific section of the project.

> **Disclaimer:** This project is for educational and portfolio purposes only. It is not financial advice, investment advice, or a recommendation to buy or sell any security.

---

## Main Features

- Automated stock and financial data collection
- Financial statement analysis
- Financial ratio calculation
- Rules-based company quality score
- Discounted Cash Flow valuation
- DCF sensitivity analysis
- Peer/comparable-company valuation
- Risk analytics
- Portfolio optimization
- Efficient frontier analysis
- Interactive Streamlit web dashboard
- Standalone Google Colab notebooks
- Automated Markdown and HTML equity research reports
- Optional SEC EDGAR integration
- Optional FRED macroeconomic data integration

---

## Project Structure

```text
Automated_Equity_Research_Platform/
├── app.py
├── notebooks/
│   ├── 00_START_HERE.ipynb
│   ├── 01_Data_Pipeline.ipynb
│   ├── 02_Financial_Analysis.ipynb
│   ├── 03_Valuation.ipynb
│   ├── 04_Risk_Analytics.ipynb
│   ├── 05_Report_Generator.ipynb
│   ├── 06_Colab_Dashboard.ipynb
│   └── 99_Combined_Research_Dashboard.ipynb
├── src/
│   └── equity_research/
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── fundamentals.py
│       ├── valuation.py
│       ├── risk.py
│       ├── visualization.py
│       ├── reporting.py
│       ├── pipeline.py
│       └── dashboard.py
├── tests/
├── data/
├── reports/
├── figures/
├── requirements.txt
├── pyproject.toml
├── README.md
├── LICENSE
└── .gitignore
```

---

## Fastest Way to Use the Project

### Option 1: Streamlit Website

The easiest user-facing version is `app.py`.

Run locally:

```bash
pip install -r requirements.txt
pip install -e .
streamlit run app.py
```

A browser window will open where users can enter:

```text
Ticker: AAPL
Peers: MSFT, GOOGL, AMZN, META
FCF Growth: 8.0%
Discount Rate: 9.0%
Terminal Growth: 2.5%
```

The app generates:

- Company overview
- Market chart
- Financial ratio analysis
- DCF valuation
- Sensitivity analysis
- Risk metrics
- Peer comparison
- Portfolio analysis
- Downloadable Markdown and HTML reports

---

## Deploy to Streamlit Community Cloud

1. Push this project to GitHub.
2. Go to Streamlit Community Cloud.
3. Select **New app**.
4. Choose your GitHub repository.
5. Set the main file path to:

```text
app.py
```

6. Deploy.

Users will then be able to use the project from a normal web link.

### Optional Streamlit Secrets

If using SEC or FRED, add these secrets in Streamlit app settings:

```toml
SEC_USER_AGENT = "Your Name your.email@example.com"
FRED_API_KEY = "your_fred_key_here"
```

---

## Google Colab Usage

Each notebook is independent. You do not need to run Notebook 1 before Notebook 2, or Notebook 2 before Notebook 3. Every notebook contains its own setup cells.

### Important GitHub setup for Colab

Before publishing, open each notebook and replace:

```text
https://github.com/YOUR_USERNAME/Automated_Equity_Research_Platform.git
```

with your real GitHub repository URL.

Example:

```text
https://github.com/abhaygadam/Automated_Equity_Research_Platform.git
```

Then someone can open any notebook from GitHub in Colab and run it.

---

## Notebook Guide

| Notebook | Purpose | Runs independently? |
|---|---|---|
| `00_START_HERE.ipynb` | Full project smoke test | Yes |
| `01_Data_Pipeline.ipynb` | Collects and cleans stock, financial, SEC, and macro data | Yes |
| `02_Financial_Analysis.ipynb` | Calculates ratios, trends, and quality score | Yes |
| `03_Valuation.ipynb` | Runs DCF and peer valuation | Yes |
| `04_Risk_Analytics.ipynb` | Calculates risk metrics and portfolio optimization | Yes |
| `05_Report_Generator.ipynb` | Generates Markdown and HTML reports | Yes |
| `06_Colab_Dashboard.ipynb` | Interactive Colab dashboard | Yes |
| `99_Combined_Research_Dashboard.ipynb` | Combined simple-access Colab dashboard | Yes |

---

## How the Platform Works

```text
User enters ticker and peers
        ↓
Data is collected
        ↓
Market features are engineered
        ↓
Financial statements are analyzed
        ↓
Ratios and quality score are calculated
        ↓
DCF and peer valuation are performed
        ↓
Risk analytics are calculated
        ↓
Portfolio optimization is performed
        ↓
Charts and tables are generated
        ↓
Final report is created
```

---

## Core Modules

| File | Description |
|---|---|
| `data.py` | Downloads market, company, SEC, and FRED data |
| `features.py` | Creates returns, volatility, moving averages, RSI, and drawdown |
| `fundamentals.py` | Calculates financial ratios and quality score |
| `valuation.py` | Runs DCF and peer comparable valuation |
| `risk.py` | Calculates risk metrics and portfolio optimization |
| `visualization.py` | Creates Plotly charts |
| `reporting.py` | Generates Markdown and HTML reports |
| `pipeline.py` | Connects the full workflow |
| `dashboard.py` | Creates the Colab widget dashboard |
| `app.py` | Creates the Streamlit web dashboard |

---

## Valuation Assumptions

The DCF model uses three major assumptions:

### FCF Growth

Expected yearly free-cash-flow growth during the forecast period.

Typical ranges:

```text
Mature company: 4–7%
Normal growth company: 7–10%
High-growth company: 10–15%
```

### Discount Rate

Represents required return and business risk.

Typical ranges:

```text
Stable company: 8–9%
Average-risk company: 9–10%
High-risk company: 10–12%
```

### Terminal Growth

Expected long-term growth after the forecast period.

Typical range:

```text
2–3%
```

Terminal growth should usually stay below the discount rate.

---

## Risk Metrics

The platform calculates:

- Annualized return
- Annualized volatility
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Historical Value at Risk
- Conditional Value at Risk
- Beta versus benchmark
- CAPM alpha
- Benchmark correlation

---

## Portfolio Optimization

When users enter peer tickers, the platform can calculate:

- Correlation matrix
- Maximum-Sharpe portfolio
- Minimum-variance portfolio
- Efficient frontier

---

## Optional SEC Integration

SEC EDGAR integration can be enabled with:

```text
SEC_USER_AGENT
```

Example:

```text
Your Name your-email@example.com
```

The SEC module retrieves official filing metadata and company facts. It supports common U.S. issuer filings and foreign private issuer filings such as `20-F` and `6-K`.

---

## Optional FRED Integration

FRED macroeconomic data can be enabled with:

```text
FRED_API_KEY
```

The project can retrieve:

- Federal Funds Rate
- 10-Year Treasury Yield
- Inflation data
- Unemployment data

---

## Example: NBIS Analysis

Inputs:

```text
Ticker: NBIS
Peers: CRWV, MSFT, GOOGL, AMZN, ORCL
FCF Growth: 8.0%
Discount Rate: 11.0%
Terminal Growth: 2.5%
```

Potential outputs:

- Stock performance chart
- Financial ratio table
- Company quality score
- Risk analytics
- Peer comparison
- Valuation output
- Full research report

If the company has negative free cash flow, the DCF may be skipped. This is intentional because forcing a DCF on negative free cash flow can create misleading results.

---

## Testing

Run:

```bash
pip install -e .[dev]
pytest
```

The included tests use synthetic data and do not require internet access.

---

## Limitations

- Yahoo Finance data can be incomplete or inconsistent.
- Financial statement labels vary across companies.
- DCF assumptions can significantly change valuation.
- Peer selection can materially affect comparable valuation.
- Historical risk metrics do not predict future risk.
- Portfolio optimization is sensitive to historical returns and covariance.
- The quality score is rules-based, not a machine-learning prediction.
- The output is educational research, not financial advice.

---

## Future Improvements

- Better automatic peer selection
- SEC XBRL as the primary statement source
- Dynamic WACC calculation
- Monte Carlo DCF valuation
- Earnings surprise analysis
- Analyst estimate integration
- More advanced portfolio constraints
- PDF report generation
- User accounts and saved research histories
- Deployed production database

---

## Skills Demonstrated

- Python programming
- Financial analysis
- Equity research
- API integration
- Data engineering
- DCF valuation
- Risk analytics
- Portfolio optimization
- Plotly visualization
- Streamlit dashboard development
- Google Colab notebook development
- Modular software architecture
- Automated report generation

---

## License

This project is licensed under the MIT License.
