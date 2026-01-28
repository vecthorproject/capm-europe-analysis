import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import io 
import requests 
import datetime
from openpyxl.styles import Alignment, Font

# =========================
# CONFIGURAZIONE PAGINA
# =========================
st.set_page_config(page_title="Analisi Beta Pro", layout="wide")

st.title("🇪🇺 Analisi Finanziaria: Focus Europa & Italia")
st.markdown("""
Strumento professionale per il calcolo del **Beta**.
**Nota Tecnica:** Per massima affidabilità, l'indice Italia utilizza il **FTSE MIB** come proxy ufficiale (High Reliability).
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
# HELPER: SESSIONI WEB
# =========================
def get_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0"
    })
    return session

# =========================
# 1. RICERCA AZIONI (YAHOO)
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
# LOGICA CALLBACK (Per aggiungere in modo sicuro)
# =========================
def add_ticker_to_portfolio():
    selection = st.session_state.get('temp_search_selection')
    if selection:
        ticker_to_add = selection[1]
        clean_name = selection[2]
        
        if ticker_to_add not in st.session_state['selected_tickers_list']:
            st.session_state['selected_tickers_list'].append(ticker_to_add)
            st.session_state['ticker_names_map'][ticker_to_add] = clean_name
        
        current_selection = st.session_state.get('multiselect_portfolio', [])
        if ticker_to_add not in current_selection:
            st.session_state['multiselect_portfolio'] = current_selection + [ticker_to_add]
            
        st.toast(f"✅ Aggiunto: {clean_name}", icon="✅")

# =========================
# DATABASE INDICI
# =========================
BENCHMARK_DICT = {
    "FTSEMIB.MI": "🇮🇹 FTSE MIB (Italia Main)",
    "^STOXX50E": "🇪🇺 Euro Stoxx 50 (Europa)",
    "^GDAXI": "🇩🇪 DAX 40 (Germania)",
    "^FCHI": "🇫🇷 CAC 40 (Francia)",
    "^GSPC": "🇺🇸 S&P 500 (USA - Global)"
}

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Configurazione")

st.sidebar.subheader("1. Cerca Titolo")
search_query = st.sidebar.text_input("Nome azienda (Invio per cercare):", "")

if search_query:
    search_results = search_yahoo_finance(search_query)
    if search_results:
        st.sidebar.selectbox(
            "Risultati trovati:", 
            options=search_results, 
            format_func=lambda x: x[0],
            key='temp_search_selection'
        )
        st.sidebar.button(
            "➕ Aggiungi al Portafoglio", 
            on_click=add_ticker_to_portfolio,
            type="primary"
        )
    else:
        st.sidebar.warning("Nessun risultato trovato.")

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Portafoglio Attivo")

if not st.session_state['selected_tickers_list']:
    st.sidebar.info("La lista è vuota.")

final_selection = st.sidebar.multiselect(
    "Gestisci titoli:",
    options=st.session_state['selected_tickers_list'],
    default=st.session_state['multiselect_portfolio'],
    format_func=lambda x: f"{st.session_state['ticker_names_map'].get(x, x)}",
    key="multiselect_portfolio" 
)

if set(final_selection) != set(st.session_state['selected_tickers_list']):
    st.session_state['selected_tickers_list'] = final_selection

st.sidebar.markdown("---")

st.sidebar.subheader("2. Parametri Analisi")
bench_display_options = list(BENCHMARK_DICT.values())
selected_bench_display = st.sidebar.selectbox("Benchmark:", bench_display_options, index=0)
benchmark_ticker = next((k for k, v in BENCHMARK_DICT.items() if v == selected_bench_display), None)

freq_option = st.sidebar.selectbox("Frequenza:", ["Settimanale (Consigliato)", "Mensile"], index=0)
interval_code = "1wk" if "Settimanale" in freq_option else "1mo"

col_d1, col_d2 = st.sidebar.columns(2)
years_back = 5 if interval_code == "1mo" else 2
default_start = datetime.date.today() - datetime.timedelta(days=365*years_back)

with col_d1: start_date = st.date_input("Data Inizio", value=default_start)
with col_d2: end_date = st.date_input("Data Fine", value=datetime.date.today())

rf_input = st.sidebar.number_input("Risk Free (BTP 10Y)", value=3.8, step=0.1) / 100
mrp_input = st.sidebar.number_input("Market Risk Premium", value=5.5, step=0.1) / 100

# =========================
# MOTORE DI CALCOLO
# =========================

def clean_dataframe_standard(df):
    if df is None or df.empty: return None
    
    df.index = pd.to_datetime(df.index).normalize()
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    
    if isinstance(df.columns, pd.MultiIndex): 
        try: df.columns = df.columns.get_level_values(0)
        except: pass
    
    price_col = None
    if "Close" in df.columns: price_col = df["Close"]
    elif "Adj Close" in df.columns: price_col = df["Adj Close"]
    elif "Ultimo" in df.columns: price_col = df["Ultimo"]
    
    if price_col is None: return None
    if isinstance(price_col, pd.DataFrame): price_col = price_col.iloc[:, 0]
    
    def get_col(name, alt_name=None):
        if name in df.columns: return df[name]
        if alt_name and alt_name in df.columns: return df[alt_name]
        return price_col 

    df_clean = pd.DataFrame({
        "Ultimo": price_col,
        "Apertura": get_col("Open", "Apertura"),
        "Massimo": get_col("High", "Massimo"),
        "Minimo": get_col("Low", "Minimo"),
        "Volume": get_col("Volume")
    })
    
    return df_clean[~df_clean.index.duplicated(keep='first')]

def resample_daily_data(df, target_interval):
    if df is None or df.empty: return None
    rule = "W-FRI" if target_interval == "1wk" else "M"
    df_res = df.resample(rule).agg({
        "Ultimo": "last", "Apertura": "first", "Massimo": "max", "Minimo": "min", "Volume": "sum"
    }).dropna()
    return df_res

def get_data_yahoo_resampled(ticker, start, end, target_interval):
    df = None
    try:
        df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    except: pass
    
    if df is None or df.empty:
        try:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(start=start, end=end, interval="1d", auto_adjust=False)
        except: pass
        
    if df is None or df.empty: return None
    df_clean = clean_dataframe_standard(df)
    if df_clean is not None:
        return resample_daily_data(df_clean, target_interval)
    return None

def get_data_pair_manager(ticker, benchmark_code, start, end, interval):
    # 1. ASSET
    df_asset = get_data_yahoo_resampled(ticker, start, end, interval)
    if df_asset is None: return None, None, f"Errore Ticker: {ticker} non risponde."
    
    # 2. BENCHMARK
    df_bench = get_data_yahoo_resampled(benchmark_code, start, end, interval)
    if df_bench is None: return None, None, f"Errore Benchmark: {benchmark_code} non risponde."

    # 3. SYNC
    if df_asset.index.tz is not None: df_asset.index = df_asset.index.tz_localize(None)
    if df_bench.index.tz is not None: df_bench.index = df_bench.index.tz_localize(None)
    df_asset = df_asset.sort_index()
    df_bench = df_bench.sort_index()
    
    merged = pd.merge_asof(
        df_asset, df_bench, left_index=True, right_index=True, 
        suffixes=('_ass', '_ben'), direction='nearest', tolerance=pd.Timedelta(days=4)
    ).dropna()
    
    if len(merged) < 5: return None, None, f"Errore Sincronizzazione: Dati insufficienti."
        
    df_asset_aligned = pd.DataFrame({"Ultimo": merged["Ultimo_ass"]})
    df_bench_aligned = pd.DataFrame({"Ultimo": merged["Ultimo_ben"]})
    bench_clean_name = BENCHMARK_DICT.get(benchmark_code, benchmark_code)
    
    return df_asset_aligned, df_bench_aligned, bench_clean_name

def calculate_metrics(df_asset, df_bench, rf, mrp, interval):
    df_asset["Var %"] = df_asset["Ultimo"].pct_change()
    df_bench["Var %"] = df_bench["Ultimo"].pct_change()
    df_asset = df_asset.dropna()
    df_bench = df_bench.dropna()
    
    common = df_asset.index.intersection(df_bench.index)
    df_asset = df_asset.loc[common]
    df_bench = df_bench.loc[common]

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
        
        # --- STILE CELLE (Centrato) ---
        center_align = Alignment(horizontal='center', vertical='center')
        header_font = Font(bold=True)
        
        # SINTESI
        summary_data = []
        for ticker, data in analysis_results.items():
            full_name = st.session_state['ticker_names_map'].get(ticker, ticker)
            summary_data.append({
                "Società": full_name, "Ticker": ticker, "Benchmark": data['bench_used_name'],
                "Beta": f"{data['stats']['Beta']:.3f}", 
                "Rendimento Atteso (CAPM)": fmt_pct(data['stats']['Exp Return'])
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name="Sintesi", index=False)
        
        # Applica stili alla sintesi
        ws_sum = writer.sheets["Sintesi"]
        for row in ws_sum.iter_rows():
            for cell in row:
                cell.alignment = center_align
                if cell.row == 1: cell.font = header_font
        
        # FOGLI INDIVIDUALI
        for ticker, data in analysis_results.items():
            sheet_name = ticker.replace(".MI", "").replace("^", "")[:28] 
            full_name = st.session_state['ticker_names_map'].get(ticker, ticker)
            bench_clean = data['bench_used_name']
            
            # HEADER (Tabellina in alto)
            metrics_df = pd.DataFrame({
                "METRICA": ["SOCIETÀ", "BETA", "COVARIANZA", "VARIANZA", "RISK FREE", "MRP", "CAPM RETURN"],
                "VALORE": [full_name, f"{data['stats']['Beta']:.4f}", f"{data['stats']['Covarianza']:.6f}",
                           f"{data['stats']['Varianza']:.6f}", fmt_pct(rf), fmt_pct(mrp), fmt_pct(data['stats']['Exp Return'])]
            })
            metrics_df.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0, index=False)
            
            # DATI
            df_asset_view = data['df_asset'][["Ultimo", "Var %"]].copy()
            df_bench_view = data['df_bench'][["Ultimo", "Var %"]].copy()
            df_asset_view.columns = [f"Prezzo", "Var %"]
            df_bench_view.columns = [f"Prezzo", "Var %"]
            
            if df_asset_view.index.tz is not None: df_asset_view.index = df_asset_view.index.tz_localize(None)
            
            # Formattazione Percentuali Dati
            for col in df_asset_view.columns:
                 if "Var" in col: df_asset_view[col] = df_asset_view[col].apply(fmt_pct)
            for col in df_bench_view.columns:
                 if "Var" in col: df_bench_view[col] = df_bench_view[col].apply(fmt_pct)
            
            # Colonna Separatore Vuota (Senza simbolo)
            df_sep = pd.DataFrame(index=df_asset_view.index)
            df_sep[" "] = "" 
            
            df_final = pd.concat([df_asset_view, df_sep, df_bench_view], axis=1)
            df_final.index.name = "Data Rilevazione"
            
            start_row_data = 12
            ws = writer.sheets[sheet_name]
            
            # INTESTAZIONI DINAMICHE
            ws.merge_cells(start_row=start_row_data-1, start_column=1, end_row=start_row_data-1, end_column=3)
            ws.cell(row=start_row_data-1, column=1, value=full_name.upper()) # Dynamic Header
            
            ws.merge_cells(start_row=start_row_data-1, start_column=5, end_row=start_row_data-1, end_column=7)
            ws.cell(row=start_row_data-1, column=5, value=bench_clean.upper()) # Dynamic Header
            
            # Scrittura Tabella Dati
            df_final.to_excel(writer, sheet_name=sheet_name, startrow=start_row_data, startcol=0)
            
            # APPLICAZIONE FORMATTAZIONE CENTRATA A TUTTO IL FOGLIO
            for row in ws.iter_rows():
                for cell in row:
                    cell.alignment = center_align
                    # Grassetto per intestazioni
                    if cell.row == 1 or cell.row == start_row_data or cell.row == start_row_data+1:
                        cell.font = header_font

            # Larghezza Colonne
            ws.column_dimensions['A'].width = 20 
            ws.column_dimensions['B'].width = 15 
            ws.column_dimensions['C'].width = 15 
            ws.column_dimensions['D'].width = 5  
            ws.column_dimensions['E'].width = 15 
            ws.column_dimensions['F'].width = 15 
            
    return output.getvalue()

# =========================
# LOGICA APP
# =========================
if st.button("🚀 Avvia Analisi (Aggiorna)", type="primary"):
    st.cache_data.clear()
    
    if not final_selection:
        st.error("⚠️ Lista titoli vuota.")
    elif start_date >= end_date:
        st.error("⚠️ Date non valide.")
    else:
        with st.spinner(f'Analisi in corso (Yahoo API)...'):
            results = {}
            error_log = []
            
            for t in final_selection:
                df_asset, df_bench, bench_name = get_data_pair_manager(t, benchmark_ticker, start_date, end_date, interval_code)
                
                if df_asset is not None:
                    res = calculate_metrics(df_asset, df_bench, rf_input, mrp_input, interval_code)
                    if res[0] is not None:
                        results[t] = {
                            "df_asset": res[0], "df_bench": res[1], "stats": res[2], "bench_used_name": bench_name
                        }
                else:
                    error_log.append(f"❌ {t}: {bench_name} (Dati insufficienti)")
            
            if results:
                st.session_state['analysis_results'] = results
                st.session_state['bench_used'] = selected_bench_display 
                st.session_state['done'] = True
                st.toast(f'✅ Analisi Completata!', icon="🚀")
            
            if error_log:
                for e in error_log: st.error(e)

# =========================
# OUTPUT
# =========================
if st.session_state.get('done'):
    results = st.session_state['analysis_results']
    
    summary_list = []
    for t, data in results.items():
        full_name = st.session_state['ticker_names_map'].get(t, t)
        summary_list.append({
            "Società": full_name, "Ticker": t, "Benchmark": data['bench_used_name'],
            "Beta": data['stats']['Beta'], "CAPM Return": f"{data['stats']['Exp Return']*100:.2f}%"
        })
    
    st.subheader("📋 Sintesi Risultati")
    st.dataframe(pd.DataFrame(summary_list), use_container_width=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Confronto Rischio")
        beta_df = pd.DataFrame(summary_list)
        fig = px.bar(beta_df, x="Società", y="Beta", text_auto=".2f", title=f"Beta vs {selected_bench_display}")
        fig.add_hline(y=1, line_dash="dash", annotation_text="Mercato")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.info(f"Dati: **{freq_option}**\n\nDal: **{start_date}**\n\nAl: **{end_date}**")

    excel_file = generate_excel_report(results, rf_input, mrp_input, selected_bench_display)
    st.download_button("📥 Scarica Report Excel", data=excel_file, file_name="Analisi_Finanziaria_Pro.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")

st.markdown("---")
with st.expander("ℹ️ Info Metodologiche"):
    st.markdown("I dati sono rettificati per dividendi e split. Il Beta è calcolato sulla covarianza dei rendimenti.")
    
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
    