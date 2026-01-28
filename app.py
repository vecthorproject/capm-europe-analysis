import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import io 
import requests 

# =========================
# CONFIGURAZIONE PAGINA
# =========================
st.set_page_config(page_title="Analisi Beta Europa", layout="wide")

st.title("🇪🇺 Analisi Finanziaria: Focus Europa")
st.markdown("""
Strumento per il calcolo del **Beta** e del **Costo del Capitale** su titoli dell'Area Euro.
I dati sono sincronizzati sui calendari di borsa europei per la massima precisione statistica.
""")

# =========================
# 1. MOTORE DI RICERCA YAHOO (Global ma consigliato EU)
# =========================
def search_yahoo_finance(query):
    if not query or len(query) < 2: return []
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=5)
        data = r.json()
        results = []
        if 'quotes' in data:
            for item in data['quotes']:
                symbol = item.get('symbol')
                name = item.get('shortname') or item.get('longname')
                exchange = item.get('exchange')
                # Filtriamo visivamente per aiutare l'utente
                if symbol and name:
                    label = f"{name} ({symbol}) - {exchange}"
                    results.append((label, symbol))
        return results
    except: return []

# =========================
# 2. RECUPERO NOMI (WIKIPEDIA)
# =========================
@st.cache_data
def get_ftse_mib_tickers_dynamic():
    STATIC_BACKUP = {
        "A2A.MI": "A2A", "ENEL.MI": "Enel", "ENI.MI": "Eni", "ISP.MI": "Intesa Sanpaolo",
        "RACE.MI": "Ferrari", "STLAM.MI": "Stellantis", "UCG.MI": "UniCredit", "TIT.MI": "TIM"
    }
    try:
        url = "https://en.wikipedia.org/wiki/FTSE_MIB"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        tables = pd.read_html(io.StringIO(response.text))
        df_list = tables[1] 
        tickers_dict = {}
        for index, row in df_list.iterrows():
            try:
                ticker = row.iloc[1] 
                name = row.iloc[0]
                if "MI" not in ticker: ticker = f"{ticker}.MI"
                tickers_dict[ticker] = name
            except: continue
        if not tickers_dict: return STATIC_BACKUP
        return tickers_dict
    except: return STATIC_BACKUP

DYNAMIC_TICKER_MAP = get_ftse_mib_tickers_dynamic()

# =========================
# DATABASE INDICI (SOLO EUROPA)
# =========================
BENCHMARK_OPTIONS = {
    "🇮🇹 FTSE MIB (Italia - Large Cap)": "FTSEMIB.MI",
    "🇮🇹 FTSE Italia All-Share (Italia - Completo)": "^FTITLMS",
    "🇪🇺 Euro Stoxx 50 (Europa - Blue Chips)": "^STOXX50E",
    "🇩🇪 DAX 40 (Germania)": "^GDAXI",
    "🇫🇷 CAC 40 (Francia)": "^FCHI"
}

# =========================
# SIDEBAR - INPUT
# =========================
st.sidebar.header("⚙️ Configurazione")

if 'selected_tickers_list' not in st.session_state:
    st.session_state['selected_tickers_list'] = ["ENEL.MI", "ISP.MI"]

st.sidebar.subheader("1. Ricerca Titoli (Europa)")
st.sidebar.caption("Cerca aziende italiane (es. *Ferrari*) o europee (es. *LVMH*, *Volkswagen*).")

# Ricerca
search_query = st.sidebar.text_input("🔍 Cerca azienda:", "")
if search_query:
    search_results = search_yahoo_finance(search_query)
    if search_results:
        selected_result = st.sidebar.selectbox("Risultati:", options=search_results, format_func=lambda x: x[0])
        if st.sidebar.button("➕ Aggiungi"):
            ticker_to_add = selected_result[1]
            if ticker_to_add not in st.session_state['selected_tickers_list']:
                st.session_state['selected_tickers_list'].append(ticker_to_add)
                st.rerun()
    else:
        st.sidebar.warning("Nessun risultato.")

# Lista Attiva
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Portafoglio Analisi")
final_selection = st.sidebar.multiselect(
    "Titoli attivi:",
    options=st.session_state['selected_tickers_list'],
    default=st.session_state['selected_tickers_list']
)

if set(final_selection) != set(st.session_state['selected_tickers_list']):
    st.session_state['selected_tickers_list'] = final_selection
    st.rerun()

st.sidebar.markdown("---")

# Parametri
st.sidebar.subheader("2. Parametri Macro")
selected_bench_name = st.sidebar.selectbox("Benchmark di Riferimento:", list(BENCHMARK_OPTIONS.keys()))
benchmark_ticker = BENCHMARK_OPTIONS[selected_bench_name]

# Nota: Il BTP è perfetto per l'Europa, il Bund sarebbe l'alternativa ma per l'Italia usiamo il BTP.
rf_input = st.sidebar.number_input("Risk Free (BTP 10Y)", value=3.8, step=0.1) / 100
mrp_input = st.sidebar.number_input("Market Risk Premium (Europa)", value=5.5, step=0.1) / 100
years_input = st.sidebar.slider("Anni (Dati Settimanali)", 1, 5, 2) 

# =========================
# MOTORE DI CALCOLO (ROBUSTO)
# =========================

@st.cache_data
def get_data_pair(ticker, benchmark, years):
    try:
        # Scarica i dati
        data = yf.download([ticker, benchmark], period=f"{years}y", interval="1wk", auto_adjust=False, progress=False)
        
        if data.empty: return None, None

        # Gestione colonne MultiIndex
        try:
            if isinstance(data.columns, pd.MultiIndex):
                closes = data["Close"]
            else:
                closes = data["Close"] if "Close" in data else data["Adj Close"]

            if ticker not in closes.columns or benchmark not in closes.columns:
                return None, None
            
            # Estrazione sicura
            def safe_extract(col_name):
                if col_name in data and isinstance(data[col_name].columns, pd.Index):
                    if ticker in data[col_name].columns and benchmark in data[col_name].columns:
                        return data[col_name][[ticker, benchmark]].dropna()
                return closes # Fallback

            opens = safe_extract("Open")
            highs = safe_extract("High")
            lows = safe_extract("Low")
            vols = safe_extract("Volume")

        except Exception: return None, None

        common_index = closes.index.dropna()
        
        df_asset = pd.DataFrame({
            "Data": common_index, 
            "Ultimo": closes[ticker], 
            "Apertura": opens[ticker] if ticker in opens else closes[ticker],
            "Massimo": highs[ticker] if ticker in highs else closes[ticker],
            "Minimo": lows[ticker] if ticker in lows else closes[ticker],
            "Volume": vols[ticker] if ticker in vols else 0
        }).set_index("Data")

        df_bench = pd.DataFrame({
            "Data": common_index, 
            "Ultimo": closes[benchmark], 
            "Apertura": opens[benchmark] if benchmark in opens else closes[benchmark],
            "Massimo": highs[benchmark] if benchmark in highs else closes[benchmark],
            "Minimo": lows[benchmark] if benchmark in lows else closes[benchmark],
            "Volume": vols[benchmark] if benchmark in vols else 0
        }).set_index("Data")
        
        return df_asset, df_bench

    except Exception: return None, None

def calculate_metrics_and_structure(df_asset, df_bench, rf, mrp):
    df_asset["Var %"] = df_asset["Ultimo"].pct_change()
    df_asset["Rendimento %"] = df_asset["Var %"]
    df_bench["Var %"] = df_bench["Ultimo"].pct_change()
    df_bench["Rendimento %"] = df_bench["Var %"]

    df_asset = df_asset.dropna()
    df_bench = df_bench.dropna()
    
    common_idx = df_asset.index.intersection(df_bench.index)
    df_asset = df_asset.loc[common_idx].sort_index(ascending=False)
    df_bench = df_bench.loc[common_idx].sort_index(ascending=False)

    if len(df_asset) < 10: return None, None, None # Troppi pochi dati

    y, x = df_asset["Var %"], df_bench["Var %"]
    covariance = np.cov(y, x)[0][1]
    variance = np.var(x, ddof=1) 
    
    if variance == 0: beta = 0 
    else: beta = covariance / variance
    
    expected_return = rf + beta * mrp
    
    stats = {"Beta": beta, "Covarianza": covariance, "Varianza": variance, "Exp Return": expected_return}
    return df_asset, df_bench, stats

def generate_excel_report(analysis_results, rf, mrp, bench_name):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        def fmt_pct(val): return f"{val * 100:.3f}%" if (not pd.isna(val) and isinstance(val, (int, float))) else val
        
        summary_data = []
        for ticker, data in analysis_results.items():
            full_name = DYNAMIC_TICKER_MAP.get(ticker, ticker)
            summary_data.append({
                "Ragione Sociale": full_name, "Ticker": ticker,
                "Beta": f"{data['stats']['Beta']:.3f}",
                "Covarianza": f"{data['stats']['Covarianza']:.6f}",
                "Varianza Mkt": f"{data['stats']['Varianza']:.6f}",
                "Rendimento Atteso (CAPM)": fmt_pct(data['stats']['Exp Return'])
            })
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Sintesi", index=False)
        
        for ticker, data in analysis_results.items():
            sheet_name = ticker.replace(".MI", "").replace("^", "")[:30] 
            full_name = DYNAMIC_TICKER_MAP.get(ticker, ticker)
            
            metrics_df = pd.DataFrame({
                "METRICA": ["SOCIETÀ", "BETA", "COVARIANZA", "VARIANZA", "RISK FREE", "MRP", "CAPM RETURN"],
                "VALORE": [full_name, f"{data['stats']['Beta']:.4f}", f"{data['stats']['Covarianza']:.6f}",
                           f"{data['stats']['Varianza']:.6f}", fmt_pct(rf), fmt_pct(mrp), fmt_pct(data['stats']['Exp Return'])]
            })
            metrics_df.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0, index=False)
            
            df_asset_view, df_bench_view = data['df_asset'].copy(), data['df_bench'].copy()
            for col in ["Var %", "Rendimento %"]:
                if col in df_asset_view.columns: df_asset_view[col] = df_asset_view[col].apply(fmt_pct)
                if col in df_bench_view.columns: df_bench_view[col] = df_bench_view[col].apply(fmt_pct)

            ws = writer.sheets[sheet_name]
            ws.cell(row=9, column=1, value=f"{full_name} ({ticker})") 
            df_asset_view.to_excel(writer, sheet_name=sheet_name, startrow=9, startcol=0)
            offset = len(df_asset_view.columns) + 1 + 2 
            ws.cell(row=9, column=offset + 1, value=f"{bench_name} (Benchmark)")
            df_bench_view.to_excel(writer, sheet_name=sheet_name, startrow=9, startcol=offset)
            
        for sheet in writer.sheets:
            ws = writer.sheets[sheet]
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = 15
            
    return output.getvalue()

# =========================
# LOGICA PRINCIPALE
# =========================
if st.button("🚀 Avvia Analisi", type="primary"):
    
    if not final_selection:
        st.error("⚠️ La lista titoli è vuota.")
    else:
        with st.spinner(f'Analisi in corso...'):
            results = {}
            for t in final_selection:
                df_asset, df_bench = get_data_pair(t, benchmark_ticker, years_input)
                if df_asset is not None and not df_asset.empty:
                    res = calculate_metrics_and_structure(df_asset, df_bench, rf_input, mrp_input)
                    if res[0] is not None:
                        results[t] = {"df_asset": res[0], "df_bench": res[1], "stats": res[2]}
            
            if results:
                st.session_state['analysis_results'] = results
                st.session_state['bench_used'] = selected_bench_name 
                st.session_state['done'] = True
                st.toast(f'✅ Completato!', icon="🚀")
            else:
                st.error(f"Nessun dato trovato. Assicurati di confrontare titoli Europei con indici Europei per la sincronizzazione delle date.")

# =========================
# VISUALIZZAZIONE
# =========================
if st.session_state.get('done'):
    results = st.session_state['analysis_results']
    bench_name = st.session_state.get('bench_used', selected_bench_name)
    
    summary_list = []
    for t, data in results.items():
        full_name = DYNAMIC_TICKER_MAP.get(t, t)
        summary_list.append({
            "Società": full_name, "Ticker": t, "Beta": data['stats']['Beta'],
            "CAPM Return": f"{data['stats']['Exp Return']*100:.2f}%"
        })
    
    st.subheader("📋 Sintesi Risultati")
    st.dataframe(pd.DataFrame(summary_list), use_container_width=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Confronto Rischio")
        beta_df = pd.DataFrame(summary_list)
        fig = px.bar(beta_df, x="Società", y="Beta", text_auto=".2f", title=f"Beta vs {bench_name} (1.0)")
        fig.add_hline(y=1, line_dash="dash", annotation_text="Mercato")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.info(f"Analisi sincronizzata su dati settimanali.")

    excel_file = generate_excel_report(results, rf_input, mrp_input, bench_name)
    st.download_button("📥 Scarica Report Excel", data=excel_file, file_name="Analisi_Finanziaria_Europea.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")

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
