import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import io 
import requests 
import datetime

# =========================
# CONFIGURAZIONE PAGINA
# =========================
st.set_page_config(page_title="Analisi Beta Pro", layout="wide")

st.title("🇪🇺 Analisi Finanziaria: Focus Europa & Italia")
st.markdown("""
Strumento professionale per il calcolo del **Beta** e del **Costo del Capitale**.
Seleziona date precise e frequenza di campionamento.
""")

# =========================
# GESTIONE STATO (SESSION STATE)
# =========================
if 'selected_tickers_list' not in st.session_state:
    st.session_state['selected_tickers_list'] = [] 

if 'ticker_names_map' not in st.session_state:
    st.session_state['ticker_names_map'] = {}

# =========================
# CONFIGURAZIONE SESSIONE (FIX PER YAHOO)
# =========================
def get_yahoo_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    })
    return session

# =========================
# 1. MOTORE DI RICERCA YAHOO
# =========================
def search_yahoo_finance(query):
    if not query or len(query) < 2: return []
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=5)
        data = r.json()
        results = []
        europe_exchanges = ['MIL', 'PAR', 'GER', 'LSE', 'AMS', 'MCE', 'LIS', 'WSE', 'VIE', 'MTA']
        
        if 'quotes' in data:
            for item in data['quotes']:
                symbol = item.get('symbol')
                name = item.get('longname') or item.get('shortname')
                exchange = item.get('exchange') 
                
                is_european = exchange in europe_exchanges or \
                              (symbol and any(symbol.endswith(s) for s in ['.MI', '.DE', '.PA', '.MC', '.AS']))
                
                if symbol and name and is_european:
                    label = f"{name} ({symbol}) - {exchange}"
                    results.append((label, symbol, name))
        return results
    except: return []

# =========================
# DATABASE INDICI
# =========================
BENCHMARK_DICT = {
    "FTSEMIB.MI": "🇮🇹 FTSE MIB (Italia - Consigliato)",
    "^FTITLMS": "🇮🇹 FTSE Italia All-Share (Italia - Completo)",
    "^STOXX50E": "🇪🇺 Euro Stoxx 50 (Europa - Blue Chips)",
    "^GDAXI": "🇩🇪 DAX 40 (Germania)",
    "^FCHI": "🇫🇷 CAC 40 (Francia)"
}

# =========================
# SIDEBAR - CONFIGURAZIONE
# =========================
st.sidebar.header("⚙️ Configurazione")

# --- 1. RICERCA TITOLI ---
st.sidebar.subheader("1. Cerca Titolo")
search_query = st.sidebar.text_input("Nome azienda (es. Enel, Ferrari):", "")

if search_query:
    search_results = search_yahoo_finance(search_query)
    if search_results:
        selected_tuple = st.sidebar.selectbox("Risultati trovati:", options=search_results, format_func=lambda x: x[0])
        
        # --- FIX BUG AGGIUNTA ---
        if st.sidebar.button("➕ Aggiungi al Portafoglio"):
            ticker_to_add = selected_tuple[1]
            clean_name = selected_tuple[2]
            
            # 1. Aggiorna la lista dati
            if ticker_to_add not in st.session_state['selected_tickers_list']:
                st.session_state['selected_tickers_list'].append(ticker_to_add)
                # 2. Aggiorna la mappa nomi
                st.session_state['ticker_names_map'][ticker_to_add] = clean_name
                
                # 3. FORZA L'AGGIORNAMENTO DEL WIDGET VISIVO SOTTOSTANTE
                st.session_state['multiselect_portfolio'] = st.session_state['selected_tickers_list']
                
                # 4. Feedback visivo immediato
                st.toast(f"✅ Aggiunto: {clean_name}", icon="💼")
                
                # 5. Ricarica la pagina per mostrare la lista aggiornata
                st.rerun()
            else:
                st.toast(f"⚠️ {clean_name} è già nel portafoglio.", icon="info")
    else:
        st.sidebar.warning("Nessun risultato europeo trovato.")

# --- 2. LISTA TITOLI ---
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Portafoglio Attivo")

if not st.session_state['selected_tickers_list']:
    st.sidebar.info("La lista è vuota. Cerca un titolo sopra.")
    
# Il widget ora è sincronizzato con la session_state tramite la chiave
final_selection = st.sidebar.multiselect(
    "Gestisci titoli:",
    options=st.session_state['selected_tickers_list'],
    default=st.session_state['selected_tickers_list'],
    key="multiselect_portfolio" # CHIAVE FONDAMENTALE PER IL FIX
)

# Se l'utente rimuove un titolo dalla lista manualmente
if set(final_selection) != set(st.session_state['selected_tickers_list']):
    st.session_state['selected_tickers_list'] = final_selection
    st.rerun()

st.sidebar.markdown("---")

# --- 3. PARAMETRI E DATE ---
st.sidebar.subheader("2. Parametri Analisi")

# A. Benchmark
bench_display_options = list(BENCHMARK_DICT.values())
selected_bench_display = st.sidebar.selectbox(
    "Benchmark di Riferimento:", 
    bench_display_options,
    index=0,
    key="bench_selector"
)
benchmark_ticker = next((k for k, v in BENCHMARK_DICT.items() if v == selected_bench_display), None)

# B. Frequenza
freq_option = st.sidebar.selectbox(
    "Frequenza Dati:",
    ["Settimanale (Consigliato)", "Mensile (Per All-Share)"],
    index=0
)
interval_code = "1wk" if "Settimanale" in freq_option else "1mo"

# C. Calendario
st.sidebar.markdown("**Periodo di Analisi:**")
col_d1, col_d2 = st.sidebar.columns(2)

years_back = 5 if interval_code == "1mo" else 2
default_start = datetime.date.today() - datetime.timedelta(days=365*years_back)

with col_d1:
    start_date = st.date_input("Data Inizio", value=default_start)
with col_d2:
    end_date = st.date_input("Data Fine", value=datetime.date.today())

if start_date >= end_date:
    st.sidebar.error("Errore: La data di inizio deve essere precedente alla fine.")

# D. Parametri Macro
rf_input = st.sidebar.number_input("Risk Free (BTP 10Y)", value=3.8, step=0.1) / 100
mrp_input = st.sidebar.number_input("Market Risk Premium", value=5.5, step=0.1) / 100

# =========================
# MOTORE DI CALCOLO
# =========================

@st.cache_data
def get_data_single_dates(ticker, start, end, interval):
    """Scarica dati con Sessione anti-blocco"""
    try:
        session = get_yahoo_session() 
        df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False, session=session)
        if df.empty: return None
        
        df.index = df.index.normalize()
        
        if "Close" in df.columns: price_col = df["Close"]
        elif "Adj Close" in df.columns: price_col = df["Adj Close"]
        else: return None 
            
        if isinstance(price_col, pd.DataFrame): price_col = price_col.iloc[:, 0]
            
        def get_col(name):
            if name in df.columns:
                c = df[name]
                if isinstance(c, pd.DataFrame): return c.iloc[:, 0]
                return c
            return price_col 

        return pd.DataFrame({
            "Ultimo": price_col,
            "Apertura": get_col("Open"),
            "Massimo": get_col("High"),
            "Minimo": get_col("Low"),
            "Volume": get_col("Volume")
        })
    except: return None

def get_data_pair_robust(ticker, benchmark, start, end, interval):
    df_asset = get_data_single_dates(ticker, start, end, interval)
    df_bench = get_data_single_dates(benchmark, start, end, interval)
    
    if df_asset is None and df_bench is None: return None, None, "Errore Totale: Dati non disponibili (Yahoo blocca la richiesta)."
    if df_asset is None: return None, None, f"Errore Ticker: {ticker} non risponde."
    if df_bench is None: return None, None, f"Errore Benchmark: {benchmark} non risponde."

    common_idx = df_asset.index.intersection(df_bench.index)
    
    min_obs = 6 if interval == "1mo" else 10
    if len(common_idx) < min_obs: 
        return None, None, f"Errore Sincronizzazione: Solo {len(common_idx)} date in comune."
        
    df_asset = df_asset.loc[common_idx].sort_index(ascending=False)
    df_bench = df_bench.loc[common_idx].sort_index(ascending=False)
    
    return df_asset, df_bench, None

def calculate_metrics(df_asset, df_bench, rf, mrp, interval):
    df_asset["Var %"] = df_asset["Ultimo"].pct_change()
    df_asset["Rendimento %"] = df_asset["Var %"]
    df_bench["Var %"] = df_bench["Ultimo"].pct_change()
    df_bench["Rendimento %"] = df_bench["Var %"]

    df_asset = df_asset.dropna()
    df_bench = df_bench.dropna()
    
    common_idx = df_asset.index.intersection(df_bench.index)
    df_asset = df_asset.loc[common_idx]
    df_bench = df_bench.loc[common_idx]

    min_obs = 5 if interval == "1mo" else 10
    if len(df_asset) < min_obs: return None, None, None 

    y, x = df_asset["Var %"], df_bench["Var %"]
    covariance = np.cov(y, x)[0][1]
    variance = np.var(x, ddof=1) 
    
    beta = covariance / variance if variance != 0 else 0
    expected_return = rf + beta * mrp
    
    stats = {"Beta": beta, "Covarianza": covariance, "Varianza": variance, "Exp Return": expected_return}
    return df_asset, df_bench, stats

def generate_excel_report(analysis_results, rf, mrp, bench_name):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        def fmt_pct(val): return f"{val * 100:.3f}%" if (not pd.isna(val) and isinstance(val, (int, float))) else val
        
        summary_data = []
        for ticker, data in analysis_results.items():
            full_name = st.session_state['ticker_names_map'].get(ticker, ticker)
            summary_data.append({
                "Ragione Sociale": full_name,
                "Ticker": ticker,
                "Beta": f"{data['stats']['Beta']:.3f}",
                "Covarianza": f"{data['stats']['Covarianza']:.6f}",
                "Varianza Mkt": f"{data['stats']['Varianza']:.6f}",
                "Rendimento Atteso (CAPM)": fmt_pct(data['stats']['Exp Return'])
            })
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Sintesi", index=False)
        
        for ticker, data in analysis_results.items():
            sheet_name = ticker.replace(".MI", "").replace("^", "")[:30] 
            full_name = st.session_state['ticker_names_map'].get(ticker, ticker)
            
            metrics_df = pd.DataFrame({
                "METRICA": ["SOCIETÀ", "BETA", "COVARIANZA", "VARIANZA", "RISK FREE", "MRP", "CAPM RETURN"],
                "VALORE": [full_name, f"{data['stats']['Beta']:.4f}", f"{data['stats']['Covarianza']:.6f}",
                           f"{data['stats']['Varianza']:.6f}", fmt_pct(rf), fmt_pct(mrp), fmt_pct(data['stats']['Exp Return'])]
            })
            metrics_df.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0, index=False)
            
            df_asset_view = data['df_asset'].copy()
            df_bench_view = data['df_bench'].copy()
            
            for col in ["Var %", "Rendimento %"]:
                if col in df_asset_view.columns: df_asset_view[col] = df_asset_view[col].apply(fmt_pct)
                if col in df_bench_view.columns: df_bench_view[col] = df_bench_view[col].apply(fmt_pct)

            ws = writer.sheets[sheet_name]
            
            ws.cell(row=9, column=1, value=f"{full_name} ({ticker})") 
            df_asset_view.to_excel(writer, sheet_name=sheet_name, startrow=9, startcol=0)
            offset = len(df_asset_view.columns) + 1 + 2 
            ws.cell(row=9, column=offset + 1, value=f"{bench_name} (Benchmark)")
            df_bench_view.to_excel(writer, sheet_name=sheet_name, startrow=9, startcol=offset)
            
            for col in ws.columns:
                max_len = 0
                for cell in col:
                    try: 
                        val_str = str(cell.value)
                        if len(val_str) > max_len: max_len = len(val_str)
                    except: pass
                
                adjusted_width = (max_len * 1.7) + 6
                if adjusted_width < 25: adjusted_width = 25
                ws.column_dimensions[col[0].column_letter].width = adjusted_width
            
    return output.getvalue()

# =========================
# LOGICA PRINCIPALE APP
# =========================
if st.button("🚀 Avvia Analisi", type="primary"):
    
    if not final_selection:
        st.error("⚠️ Inserisci almeno un titolo.")
    elif start_date >= end_date:
        st.error("⚠️ Controlla le date.")
    else:
        with st.spinner(f'Analisi in corso dal {start_date} al {end_date} ({freq_option})...'):
            results = {}
            error_log = []
            
            for t in final_selection:
                df_asset, df_bench, err_msg = get_data_pair_robust(t, benchmark_ticker, start_date, end_date, interval_code)
                
                if df_asset is not None:
                    res = calculate_metrics(df_asset, df_bench, rf_input, mrp_input, interval_code)
                    if res[0] is not None:
                        results[t] = {"df_asset": res[0], "df_bench": res[1], "stats": res[2]}
                else:
                    error_log.append(f"❌ {t}: {err_msg}")
            
            if results:
                st.session_state['analysis_results'] = results
                st.session_state['bench_used'] = selected_bench_display 
                st.session_state['done'] = True
                st.toast(f'✅ Completato!', icon="🚀")
                if error_log:
                    with st.expander("⚠️ Avvisi sui dati"):
                        for e in error_log: st.write(e)
            else:
                st.error("Nessun dato valido trovato nel periodo selezionato.")
                if error_log:
                    for e in error_log: st.write(e)

# =========================
# VISUALIZZAZIONE RISULTATI
# =========================
if st.session_state.get('done'):
    results = st.session_state['analysis_results']
    bench_name = st.session_state.get('bench_used', selected_bench_display)
    
    summary_list = []
    for t, data in results.items():
        full_name = st.session_state['ticker_names_map'].get(t, t)
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
        st.info(f"Dati: **{freq_option}**\n\nDal: **{start_date}**\n\nAl: **{end_date}**")

    excel_file = generate_excel_report(results, rf_input, mrp_input, bench_name)
    st.download_button("📥 Scarica Report Excel", data=excel_file, file_name="Analisi_Finanziaria.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")

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
