import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import io
import requests
import datetime
import json

# =========================
# CONFIGURAZIONE PAGINA
# =========================
st.set_page_config(page_title="Analisi Beta Pro", layout="wide")

st.title("🇪🇺 Analisi Finanziaria: Focus Europa & Italia")
st.markdown("""
Strumento professionale per il calcolo del **Beta** e del **Costo del Capitale**.
**Sistema Blindato:** Include 3 livelli di recupero dati (Libreria -> API Diretta -> Provider Alternativo).
""")

# =========================
# GESTIONE STATO
# =========================
if 'selected_tickers_list' not in st.session_state:
    st.session_state['selected_tickers_list'] = []
if 'ticker_names_map' not in st.session_state:
    st.session_state['ticker_names_map'] = {}
if 'multiselect_portfolio' not in st.session_state:
    st.session_state['multiselect_portfolio'] = []

# =========================
# HELPER: SESSIONI & API DIRETTA
# =========================
def get_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    })
    return session

def download_yahoo_direct_json(ticker, interval, period_str):
    try:
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?interval={interval}&range={period_str}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        data = r.json()

        if data.get("chart", {}).get("result"):
            res = data["chart"]["result"][0]
            ts = res["timestamp"]
            q = res["indicators"]["quote"][0]

            df = pd.DataFrame({
                "Ultimo": q["close"],
                "Apertura": q["open"],
                "Massimo": q["high"],
                "Minimo": q["low"],
                "Volume": q["volume"]
            }, index=pd.to_datetime(ts, unit="s"))

            return df
    except:
        return None
    return None

# =========================
# MOTORE DI RICERCA
# =========================
def search_yahoo_finance(query):
    if not query or len(query) < 2:
        return []
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        data = r.json()
        results = []

        europe_exchanges = ['MIL', 'PAR', 'GER', 'LSE', 'AMS', 'MCE', 'LIS', 'WSE', 'VIE', 'MTA']

        for item in data.get("quotes", []):
            symbol = item.get("symbol")
            name = item.get("longname") or item.get("shortname")
            exchange = item.get("exchange")

            if symbol and name and (
                exchange in europe_exchanges or
                symbol.endswith(('.MI', '.DE', '.PA', '.MC', '.AS'))
            ):
                results.append((f"{name} ({symbol}) - {exchange}", symbol, name))
        return results
    except:
        return []

# =========================
# DATABASE INDICI
# =========================
BENCHMARK_DICT = {
    "FTSEMIB.MI": "🇮🇹 FTSE MIB (Yahoo Finance)",
    "STOOQ_ALLSHARE": "🇮🇹 FTSE Italia All-Share (Fonte: Stooq)",
    "^STOXX50E": "🇪🇺 Euro Stoxx 50 (Europa)",
    "^GDAXI": "🇩🇪 DAX 40 (Germania)",
    "^FCHI": "🇫🇷 CAC 40 (Francia)"
}

# =========================
# CLEAN DATAFRAME
# =========================
def clean_dataframe(df):
    if df is None or df.empty:
        return None

    df.index = pd.to_datetime(df.index).tz_localize(None)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    price = df["Adj Close"] if "Adj Close" in df.columns else df.get("Close") or df.get("Ultimo")
    if price is None:
        return None

    return pd.DataFrame({
        "Ultimo": price,
        "Apertura": df["Open"] if "Open" in df.columns else price,
        "Massimo": df["High"] if "High" in df.columns else price,
        "Minimo": df["Low"] if "Low" in df.columns else price,
        "Volume": df["Volume"] if "Volume" in df.columns else pd.Series(0, index=df.index)
    }).dropna()

# =========================
# DATA SMART
# =========================
def get_data_yahoo_smart(ticker, start, end, interval):
    session = get_session()
    years = max(1, int((end - start).days / 365) + 1)
    period_str = f"{min(years, 10)}y"

    try:
        df = yf.Ticker(ticker, session=session).history(
            start=start, end=end, interval=interval, auto_adjust=False
        )
        clean = clean_dataframe(df)
        if clean is not None:
            return clean
    except:
        pass

    try:
        df = yf.download(
            ticker, start=start, end=end, interval=interval,
            auto_adjust=False, progress=False, session=session
        )
        clean = clean_dataframe(df)
        if clean is not None:
            return clean
    except:
        pass

    df = download_yahoo_direct_json(ticker, interval, period_str)
    if df is not None:
        df = df.loc[(df.index >= start) & (df.index <= end)]
        return clean_dataframe(df)

    return None

# =========================
# CALCOLO METRICHE
# =========================
def calculate_metrics(df_asset, df_bench, rf, mrp, interval):
    df = pd.concat([
        df_asset["Ultimo"].pct_change(),
        df_bench["Ultimo"].pct_change()
    ], axis=1).dropna()

    min_obs = 24 if interval == "1wk" else 36
    if len(df) < min_obs:
        return None, None, None

    cov = df.cov().iloc[0, 1]
    var = df.iloc[:, 1].var()

    beta = cov / var if var != 0 else np.nan
    exp_ret = rf + beta * mrp

    df_asset["Var %"] = df.iloc[:, 0]
    df_bench["Var %"] = df.iloc[:, 1]

    return df_asset, df_bench, {
        "Beta": beta,
        "Covarianza": cov,
        "Varianza": var,
        "Exp Return": exp_ret
    }

# =========================
# RELAZIONE METODOLOGICA (Espansa e Aggiornata)
# =========================
st.markdown("---")
st.header("📚 Documentazione Tecnica e Metodologica")

with st.expander("📖 Leggi la Relazione Completa (Metodologia, Indici e Formule)"):
    st.markdown(r"""
    ### 1. Scopo dell'Analisi
    Questo strumento è progettato per stimare il **Costo del Capitale Proprio (Ke)** e il **Rischio Sistematico (Beta)** di un portafoglio di titoli, confrontandoli dinamicamente con un indice di mercato (Benchmark) a scelta dell'utente.
    
    L'output finale è un report Excel professionale strutturato "Side-by-Side" (fianco a fianco), che permette una verifica puntuale della correlazione tra il singolo titolo e l'indice scelto settimana per settimana.

    ---

    ### 2. Selezione del Benchmark (Indici Italiani ed Internazionali)
    Il modello non è limitato al mercato domestico, ma permette di scegliere tra diversi scenari di confronto a seconda della natura del titolo analizzato:
    
    #### A. Mercato Italia (Focus Domestico)
    * **FTSE MIB:** Rappresenta le **40 società italiane** a maggiore capitalizzazione (Blue Chips, ~80% del mercato).
        * *Uso:* Benchmark standard per titoli grandi e liquidi (es. Enel, Intesa, Eni).
    * **FTSE Italia All-Share:** Include FTSE MIB + Mid Cap + Small Cap.
        * *Uso:* Preferibile per analizzare titoli a media/piccola capitalizzazione, per un confronto con l'economia reale italiana più ampia.

    #### B. Mercati Internazionali (Confronto Globale)
    È possibile selezionare indici esteri per valutare la sensibilità di titoli multinazionali:
    * **S&P 500 (USA):** Per confrontare titoli italiani quotati anche a New York o con forte export in America (es. Ferrari, Tenaris).
    * **DAX (Germania) / CAC 40 (Francia):** Per confronti settoriali europei.
    * **Euro Stoxx 50:** Per valutare il titolo nel contesto dei leader europei.
    
    ---

    ### 3. Struttura dei Dati e Timeframe
    * **Frequenza:** Dati **Settimanali (Weekly)**.
        * *Motivazione:* Su un orizzonte di breve/medio periodo (2-5 anni), i dati settimanali offrono il miglior compromesso tra numero di osservazioni (statistica robusta, >100 obs) e riduzione del "rumore" (volatilità giornaliera eccessiva).
    * **Layout Excel:** Il file generato espone i dati in modalità **Side-by-Side**: le colonne del Titolo sono affiancate a quelle dell'Indice selezionato (che cambia dinamicamente in base alla scelta dell'utente).

    ---

    ### 4. Il Motore Matematico (Calcolo del Beta)
    Il tool calcola il **Beta ($\beta$)** replicando esplicitamente la metodologia accademica classica basata sui rendimenti:

    $$ \beta = \frac{Cov(R_{asset}, R_{benchmark})}{Var(R_{benchmark})} $$

    Dove:
    * **$Cov$:** Covarianza tra i rendimenti settimanali del titolo e dell'indice scelto.
    * **$Var$:** Varianza dei rendimenti settimanali dell'indice scelto.
    * **$R$ (Rendimento):** Calcolato come Variazione Percentuale Semplice ($\frac{P_t - P_{t-1}}{P_{t-1}}$).

    ---

    ### 5. Costo del Capitale (CAPM)
    Il rendimento atteso finale (**Expected Return**) è calcolato secondo il **Capital Asset Pricing Model**:

    $$ E(R) = R_f + \beta \times (R_m - R_f) $$

    I parametri macroeconomici utilizzati sono specifici per l'investitore domestico:
    * **Risk-Free Rate ($R_f$):** Rendimento del **BTP Italia a 10 Anni** (Default impostato a ~3.8%). Si preferisce il BTP al Bund tedesco per incorporare il rischio-paese reale sopportato dall'investitore italiano.
    * **Market Risk Premium ($MRP$):** Fissato al **5.5%**.
        * *Fonte:* **Survey IESE Business School (Pablo Fernandez, 2025)**. Rappresenta il premio per il rischio azionario medio richiesto dagli investitori istituzionali per il mercato italiano.
    """)
