# Stock analysis and portfolio builder

A web app that analyses the stocks you type in, values them three different ways, reads the latest
news and public sentiment, and builds an optimal portfolio. It runs in your browser with one command.

> **Educational tool, not financial advice.** Past performance does not guarantee future results.
> Free data can be delayed, incomplete or wrong. Valuation results depend heavily on assumptions.

## What you get

| Tab | What it does |
|---|---|
| Overview | One card per stock (price, rating, fair value, upside, sentiment) and a portfolio headline |
| Price & Risk | Price with 50/200-day averages, RSI, MACD, Bollinger Bands, Sharpe, drawdown, beta, VaR, correlations |
| News & Sentiment | Latest headlines and social posts, sentiment score per item, spikes in chatter, key event headlines |
| Ratios | 33 ratios, colour-coded, with plain-English explanations and CSV/Excel download |
| Valuation | Comparative (peer multiples), DCF with WACC, sensitivity and scenarios, and asset-based valuation |
| Recommendations | Score out of 100 from six pillars, a rating, and the reasons in plain English |
| Portfolio Optimiser | Efficient frontier, optimal weights, risk parity, backtest, Monte Carlo, your own weights, Excel export |
| Stock Screener | Screens the S&P 100 and FTSE 100 and suggests candidates you can add with one click |
| Help & Glossary | How to use the app and every metric explained |

## Folder guide

```
stock_app/
├── app.py                 the web app (user interface only)
├── config.py              default settings and constants (edit to change defaults)
├── requirements.txt       the libraries to install
├── .env.example           template for the optional API keys
├── data/                  downloading prices, statements, news and social posts
├── analysis/              all calculations (ratios, valuation, risk, sentiment, scoring, portfolio)
├── ui/                    charts (Plotly) and reusable page components
└── tests/test_core.py     checks that the key calculations are right
```

---

## Setup for a complete beginner

### 1. Install Python and check the version

You need **Python 3.10 or newer**.

* **Windows:** download it from <https://www.python.org/downloads/>. In the installer, **tick "Add python.exe to PATH"**.
  Then open *Command Prompt* (press the Windows key, type `cmd`, press Enter) and type:
  ```
  python --version
  ```
* **Mac:** download it from <https://www.python.org/downloads/> (or use `brew install python`). Open *Terminal* and type:
  ```
  python3 --version
  ```

It should print something like `Python 3.12.3`. If it says 3.9 or lower, install a newer version.

### 2. Open a terminal inside the project folder

Unzip `stock_app.zip` somewhere you can find it, such as your Documents folder.

* **Windows:** open the `stock_app` folder in File Explorer, click the address bar at the top, type `cmd` and press Enter.
* **Mac:** open Terminal, type `cd ` (with a space after it), drag the `stock_app` folder into the Terminal window, press Enter.

### 3. Create and activate a virtual environment

A *virtual environment* is an isolated folder of libraries, so this project does not clash with anything else on your computer.

**Windows**
```
python -m venv venv
venv\Scripts\activate
```
If PowerShell refuses to activate it, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` and try again.

**Mac**
```
python3 -m venv venv
source venv/bin/activate
```

You should now see `(venv)` at the start of the line. **Repeat only the activate line** (not the first line) each time you open a new terminal.

### 4. Install the libraries

```
pip install -r requirements.txt
```

`pip` is Python's package installer. This takes a few minutes. If the `PyPortfolioOpt` line fails, delete that one line from
`requirements.txt` and run the command again: the app automatically uses SciPy instead.

### 5. (Optional) Get free API keys

**Every key is optional.** With no keys the app uses Yahoo Finance, Stooq, Google News, Yahoo Finance RSS and StockTwits.
Keys add extra sources. To use them, copy the template to a file called `.env` and paste your keys into it:

* Windows: `copy .env.example .env`   Mac: `cp .env.example .env`
* Open `.env` in Notepad or TextEdit, paste each key straight after the `=` sign (no quotes, no spaces), and save.
* Restart the app after changing `.env`. Never share this file or upload it to GitHub.

| Key | How to get it | Paste it as |
|---|---|---|
| **Reddit** (posts from r/stocks, r/investing, r/wallstreetbets and more) | Log in at reddit.com, open <https://www.reddit.com/prefs/apps>, click **create another app**, choose **script**, give it any name, set redirect uri to `http://localhost:8080`, click create. The **client ID** is the short code under the app name; the **secret** is shown as "secret". Reddit has tightened API access for new apps and may ask you to request approval first. If you cannot get a key, just skip it. | `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, and `REDDIT_USER_AGENT` (any text, e.g. `stock-app/1.0 by u/yourname`) |
| **Finnhub** (extra news, US companies) | Sign up free at <https://finnhub.io>. Your key is on the dashboard. | `FINNHUB_API_KEY` |
| **NewsAPI** (extra news) | Register free at <https://newsapi.org/register>. The free plan is for local testing and delays articles by up to a day, so it may not work once deployed online. | `NEWSAPI_KEY` |
| **Alpha Vantage** (backup prices) | Claim a free key at <https://www.alphavantage.co/support/#api-key>. Only used if Yahoo and Stooq both fail; about 25 requests a day. | `ALPHAVANTAGE_API_KEY` |
| **X / Twitter** | The official API needs a paid plan with search access. If you have a Bearer token, paste it. If not, leave it empty: X is skipped silently. | `X_BEARER_TOKEN` |

### 6. Run the app

```
streamlit run app.py
```

Your browser opens at **<http://localhost:8501>**. If it does not, type that address into your browser. Press `Ctrl+C` in the terminal to stop the app.

### 7. First steps in the app

1. The sidebar already has example tickers. Edit them if you like (separate with commas or spaces; add a suffix for non-US stocks: `VOD.L` London, `ASML.AS` Amsterdam, `SAP.DE` Germany, `7203.T` Tokyo).
2. Press **Run analysis**. The first run takes a minute or two because it downloads prices, financial statements and news. Settings in the sidebar take effect when you press this button.
3. Explore the tabs. Open **Advanced settings** in the sidebar to change the equity risk premium, terminal growth, recommendation weights and more.
4. In the Valuation tab you can override every DCF and WACC assumption with sliders; ratings update instantly.
5. **Dark or light mode:** use the menu at the top right, then Settings, then Theme. Charts and tables follow it.

---

## Common errors and fixes

| What you see | What it means and how to fix it |
|---|---|
| `ModuleNotFoundError: No module named 'streamlit'` (or any other module) | The libraries are not installed in the active environment. Activate the virtual environment (step 3) and run `pip install -r requirements.txt` again. |
| `'python' is not recognized` / `command not found` | Python is not on your PATH. On Windows reinstall Python and tick "Add python.exe to PATH". On Mac use `python3` instead of `python`. |
| `'pip' is not recognized` | Use `python -m pip install -r requirements.txt` (Mac: `python3 -m pip ...`). |
| "Rate limited" or "Too Many Requests" | Yahoo Finance limits how fast data can be requested. The app already retries politely and caches results for 15 minutes. Wait a few minutes, use fewer tickers, and avoid pressing Run analysis repeatedly. |
| "No price data was found for: XYZ" | The ticker is misspelt or needs an exchange suffix (VOD.L, ASML.AS). Search the company on finance.yahoo.com and copy the exact symbol. |
| Many ratios show N/A | Yahoo does not provide full statements for every stock, especially smaller or non-US companies. Banks also lack several ratios by nature. |
| Errors mentioning `yfinance` or `JSONDecodeError` | Yahoo changed something. Update the library: `pip install -U yfinance`. |
| `PyPortfolioOpt` or `cvxpy` fails to install | Delete the `PyPortfolioOpt` line in `requirements.txt` and reinstall. The app uses SciPy and tells you which solver it used. |
| "Port 8501 is already in use" | Another copy is running. Close it or use `streamlit run app.py --server.port 8502`. |
| Sentiment says "VADER not installed" | Run `pip install vaderSentiment`. The app still works using a built-in word list. |
| Anything else | Open `stock_app.log` (created next to `app.py`): it records the technical details of every problem. |

## Run the tests

The tests need no internet connection. From the `stock_app` folder with the environment active:

```
pytest
```

or, without pytest: `python tests/test_core.py`. All tests should pass.

## Optional: FinBERT sentiment

FinBERT is a finance-trained AI language model that scores headlines more accurately than the default VADER word list.
It is a large download (about 1 GB). Install it with `pip install transformers torch`, then choose it under
**Advanced settings, Sentiment model** in the sidebar. If it is not installed the app quietly uses VADER.

## Deploy it free online (Streamlit Community Cloud)

1. Create a free account at <https://github.com> and make a new repository (private is fine).
2. Upload the project files to it. The included `.gitignore` keeps your `.env` file and log out of the upload. **Never upload your keys.**
3. Go to <https://share.streamlit.io>, sign in with GitHub and click **Create app**.
4. Choose your repository and branch, and set **Main file path** to `app.py`.
5. Click **Advanced settings**. Under **Secrets**, paste your keys (only the ones you have) in this format, then save:
   ```
   REDDIT_CLIENT_ID = "your-id"
   REDDIT_CLIENT_SECRET = "your-secret"
   REDDIT_USER_AGENT = "stock-app/1.0 by u/yourname"
   FINNHUB_API_KEY = "your-key"
   ```
   The app reads these automatically in place of the `.env` file. Choose Python 3.11 or 3.12 if asked.
6. Click **Deploy**. If `PyPortfolioOpt` fails to build there, remove that line from `requirements.txt` and push again.

Note that on a shared cloud host Yahoo Finance may rate-limit requests more often than on your own computer.

## How the numbers are built (short version)

* **Returns and risk** use adjusted close prices (corrected for splits and dividends), 252 trading days a year, converted into your base currency.
* **Valuation:** comparative uses peer medians of P/E, EV/EBITDA, P/S and P/B; DCF discounts forecast free cash flow to the firm at the WACC, with a Gordon-growth and an exit-multiple terminal value;
  asset-based applies adjustable haircuts to the balance sheet. The blended fair value defaults to 50% DCF, 35% comparative, 15% asset-based.
* **Score and rating:** Valuation 30%, Quality 20%, Growth 15%, Momentum 15%, Risk 10%, Sentiment 10%. Strong Buy 80+, Buy 65-79, Hold 45-64, Reduce 30-44, Sell below 30.
* **Portfolio:** Markowitz mean-variance optimisation with Ledoit-Wolf covariance shrinkage. The backtest fits weights on the first 70% of the data and tests on the last 30% to avoid look-ahead bias.

The **Help & Glossary** tab explains every metric in plain English.
# Stock-and-Portfolio-Optimiser-Test
# Stock-and-Portfolio-Optimiser-Test
# Stock-and-Portfolio-Optimiser-Test
