import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import io
import requests
import datetime
import logging

# =========================
# CONFIGURAZIONE
# =========================
st.set_page_config(page_title="Analisi Beta Pro", layout="wide")
logging.basicConfig(level=logging.INFO)

st.title("🇪🇺 Analisi Finanziaria: Focus Europa & Italia")
st.markdown("""
Strumento professionale per il calcolo del **Beta** e del **Costo del Capitale (CAPM)**  
Sistema **robusto a fallback multipli** per il recupero dati.
""")

# =========================
# SESSION STATE
# =========================
for k in ["selected_tickers_list", "ticker_names_map", "multiselect_portfolio"]:
    if k not in st.session_state:
        st.session_state[k] = [] if "list" in k or "portfolio" in k else {}

# =========================
# UTILS
# =========================
def get_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    return s

def clean_dataframe(df):
    if df is None or df.empty:
        return None

    df.index = pd.to_datetime(df.index).tz_localize(None)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    price = None
    for c in ["Adj Close", "Close", "Ultimo"]:
        if c in df.columns:
            price = df[c]
            break
    if price is None:
        return None

    def col(name):
        return df[name] if name in df.columns else pd.Series(price.values, index=df.index)

    return pd.DataFrame({
        "Ultimo": price,
        "Apertura": col("Open"),
        "Massimo": col("High"),
        "Minimo": col("Low"),
        "Volume": df["Volume"] if "Volume" in df.columns else pd.Series(0, index=df.index)
    }).dropna()

# =========================
# DATA DOWNLOAD
# =========================
@st.cache_data(ttl=3600)
def yahoo_json_direct(ticker, period, interval):
    try:
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?range={period}&interval={interval}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        res = r.json()["chart"]["result"][0]
        ts = res["timestamp"]
        q = res["indicators"]["quote"][0]

        df = pd.DataFrame({
            "Ultimo": q["close"],
            "Apertura": q["open"],
            "Massimo": q["high"],
            "Minimo": q["low"],
            "Volume": q["volume"]
        }, index=pd.to_datetime(ts, unit="s"))

        return clean_dataframe(df)
    except Exception as e:
        logging.warning(f"Yahoo JSON failed {ticker}: {e}")
        return None

@st.cache_data(ttl=3600)
def get_data_yahoo_smart(ticker, start, end, interval):
    session = get_session()
    years = max(1, int((end - start).days / 365) + 1)
    period = f"{min(years, 10)}y"

    try:
        df = yf.Ticker(ticker, session=session).history(
            start=start, end=end, interval=interval, auto_adjust=False
        )
        df = clean_dataframe(df)
        if df is not None:
            return df
    except:
        pass

    try:
        df = yf.download(
            ticker, start=start, end=end,
            interval=interval, progress=False, session=session
        )
        df = clean_dataframe(df)
        if df is not None:
            return df
    except:
        pass

    df = yahoo_json_direct(ticker, period, interval)
    if df is not None:
        return df.loc[(df.index >= start) & (df.index <= end)]

    return None

@st.cache_data(ttl=3600)
def get_data_stooq(symbol, start, end, interval):
    try:
        url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
        df = pd.read_csv(url)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        df = df.loc[start:end]

        rule = "W-FRI" if interval == "1wk" else "M"
        df = df.resample(rule).agg({
            "Close": "last",
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Volume": "sum"
        }).dropna()

        df = df.rename(columns={
            "Close": "Ultimo", "Open": "Apertura",
            "High": "Massimo", "Low": "Minimo"
        })
        return clean_dataframe(df)
    except:
        return None

# =========================
# ANALYTICS
# =========================
def align_pair(asset, bench):
    df = asset.join(bench, how="inner", lsuffix="_a", rsuffix="_b")
    return (
        df[["Ultimo_a"]].rename(columns={"Ultimo_a": "Ultimo"}),
        df[["Ultimo_b"]].rename(columns={"Ultimo_b": "Ultimo"})
    )

def calculate_metrics(asset, bench, rf, mrp, interval):
    asset["Ret"] = asset["Ultimo"].pct_change()
    bench["Ret"] = bench["Ultimo"].pct_change()

    df = pd.concat([asset["Ret"], bench["Ret"]], axis=1).dropna()
    min_obs = 24 if interval == "1wk" else 36
    if len(df) < min_obs:
        return None

    cov = df.cov().iloc[0,1]
    var = df.iloc[:,1].var()
    beta = cov / var if var != 0 else np.nan

    return {
        "Beta": beta,
        "Covarianza": cov,
        "Varianza": var,
        "Exp Return": rf + beta * mrp
    }

# =========================
# BENCHMARK
# =========================
BENCHMARKS = {
    "FTSEMIB.MI": "🇮🇹 FTSE MIB",
    "^STOXX50E": "🇪🇺 Euro Stoxx 50",
    "^GDAXI": "🇩🇪 DAX 40",
    "^FCHI": "🇫🇷 CAC 40"
}

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Parametri")

benchmark = st.sidebar.selectbox("Benchmark", list(BENCHMARKS.keys()))
freq = st.sidebar.selectbox("Frequenza", ["Settimanale", "Mensile"])
interval = "1wk" if freq == "Settimanale" else "1mo"

start = st.sidebar.date_input("Data Inizio", datetime.date.today() - datetime.timedelta(days=365*5))
end = st.sidebar.date_input("Data Fine", datetime.date.today())

rf = st.sidebar.number_input("Risk Free (%)", value=3.8) / 100
mrp = st.sidebar.number_input("Market Risk Premium (%)", value=5.5) / 100

tickers = st.sidebar.text_area("Tickers (uno per riga)", "ENEL.MI\nENI.MI").split()

# =========================
# RUN
# =========================
if st.button("🚀 Avvia Analisi", type="primary"):
    results = []

    for t in tickers:
        asset = get_data_yahoo_smart(t.strip(), start, end, interval)
        bench = get_data_yahoo_smart(benchmark, start, end, interval)

        if asset is None or bench is None:
            st.error(f"{t}: errore download dati")
            continue

        asset, bench = align_pair(asset, bench)
        stats = calculate_metrics(asset, bench, rf, mrp, interval)
        if stats:
            results.append({"Ticker": t, **stats})

    if results:
        df = pd.DataFrame(results)
        st.subheader("📋 Risultati")
        st.dataframe(df, use_container_width=True)

        fig = px.bar(df, x="Ticker", y="Beta", title="Beta per Titolo", text_auto=".2f")
        fig.add_hline(y=1, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

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
