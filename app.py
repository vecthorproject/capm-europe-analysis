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
st.set_page_config(page_title="Analisi Finanziaria Pro", layout="wide")

st.title("📊 Analisi Finanziaria: Beta, CAPM e Dati Storici")
st.markdown("""
Strumento professionale per il calcolo del **Costo del Capitale** e del **Rischio Sistematico**.
Ricerca titoli intelligente (Global Search) e confronto Side-by-Side.
""")

# =========================
# 1. MOTORE DI RICERCA YAHOO (NUOVO)
# =========================
def search_yahoo_finance(query):
    """
    Cerca il ticker su Yahoo Finance partendo dal nome dell'azienda.
    Restituisce una lista di opzioni trovate.
    """
    if not query or len(query) < 2: return []
    
    try:
        # API "nascosta" di Yahoo Finance per l'autocomplete
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        r = requests.get(url, headers=headers, timeout=5)
        data = r.json()
        
        results = []
        if 'quotes' in data:
            for item in data['quotes']:
                # Filtriamo solo le azioni (Equity) per evitare ETF o futures strani se non richiesti
                # Ma lasciamo aperto per flessibilità.
                symbol = item.get('symbol')
                name = item.get('shortname') or item.get('longname')
                exchange = item.get('exchange')
                
                if symbol and name:
                    # Formattiamo per la visualizzazione: "Ferrari N.V. (RACE.MI) - Milan"
                    label = f"{name} ({symbol}) - {exchange}"
                    results.append((label, symbol))
                    
        return results # Restituisce lista di tuple (Etichetta Visibile, Ticker Reale)
        
    except Exception as e:
        return []

# =========================
# 2. RECUPERO DINAMICO FTSE MIB (WIKIPEDIA)
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
# DATABASE INDICI
# =========================
BENCHMARK_OPTIONS = {
    "FTSE MIB (Italia 40)": "FTSEMIB.MI",
    "FTSE Italia All-Share": "^FTITLMS",
    "DAX (Germania)": "^GDAXI",
    "S&P 500 (USA)": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Euro Stoxx 50": "^STOXX50E"
}

# =========================
# SIDEBAR - INPUT INTELLIGENTE
# =========================
st.sidebar.header("⚙️ Configurazione")

# --- GESTIONE LISTA TITOLI ---
if 'selected_tickers_list' not in st.session_state:
    st.session_state['selected_tickers_list'] = ["ENEL.MI", "ISP.MI"] # Default iniziale

st.sidebar.subheader("1. Ricerca e Aggiungi Titoli")

# A. BARRA DI RICERCA INTELLIGENTE
search_query = st.sidebar.text_input("🔍 Cerca azienda (es. Ferrari, Apple)", "")
search_results = []

if search_query:
    search_results = search_yahoo_finance(search_query)

if search_results:
    # Mostra i risultati in una selectbox
    selected_result = st.sidebar.selectbox(
        "Risultati trovati:", 
        options=search_results, 
        format_func=lambda x: x[0] # Mostra l'etichetta bella
    )
    
    # Bottone per aggiungere
    if st.sidebar.button("➕ Aggiungi alla lista"):
        ticker_to_add = selected_result[1] # Prende il ticker reale (es. RACE.MI)
        if ticker_to_add not in st.session_state['selected_tickers_list']:
            st.session_state['selected_tickers_list'].append(ticker_to_add)
            st.success(f"Aggiunto: {ticker_to_add}")
            st.rerun() # Ricarica la pagina per aggiornare la lista sotto

# B. VISUALIZZA E GESTISCI LA LISTA ATTUALE
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Il tuo Portafoglio Analisi")

# Multiselect che funge da "Cestino" (puoi rimuovere i titoli cliccando la X)
final_selection = st.sidebar.multiselect(
    "Titoli attivi:",
    options=st.session_state['selected_tickers_list'],
    default=st.session_state['selected_tickers_list']
)

# Aggiorniamo lo stato se l'utente rimuove qualcosa dalla multiselect
if set(final_selection) != set(st.session_state['selected_tickers_list']):
    st.session_state['selected_tickers_list'] = final_selection
    st.rerun()

st.sidebar.markdown("---")

# 2. Benchmark e Parametri
st.sidebar.subheader("2. Parametri")
selected_bench_name = st.sidebar.selectbox("Benchmark:", list(BENCHMARK_OPTIONS.keys()))
benchmark_ticker = BENCHMARK_OPTIONS[selected_bench_name]

rf_input = st.sidebar.number_input("Risk Free (BTP 10Y)", value=3.8, step=0.1) / 100
mrp_input = st.sidebar.number_input("Market Risk Premium", value=5.5, step=0.1) / 100
years_input = st.sidebar.slider("Anni", 1, 5, 2) 

st.sidebar.info(f"Titoli pronti: {len(final_selection)}. Premi Avvia.")

# =========================
# MOTORE DI CALCOLO
# =========================

@st.cache_data
def get_data_pair(ticker, benchmark, years):
    try:
        data = yf.download([ticker, benchmark], period=f"{years}y", interval="1wk", auto_adjust=False, progress=False)
        try:
            closes = data["Close"][[ticker, benchmark]].dropna()
            # Se scarica solo Close è un problema, servono anche gli altri per l'excel
            opens = data["Open"][[ticker, benchmark]].dropna() if "Open" in data else closes
            highs = data["High"][[ticker, benchmark]].dropna() if "High" in data else closes
            lows = data["Low"][[ticker, benchmark]].dropna() if "Low" in data else closes
            vols = data["Volume"][[ticker, benchmark]].dropna() if "Volume" in data else closes
        except: return None, None

        common_index = closes.index
        df_asset = pd.DataFrame({
            "Data": common_index, "Ultimo": closes[ticker], "Apertura": opens[ticker],
            "Massimo": highs[ticker], "Minimo": lows[ticker], "Volume": vols[ticker]
        }).set_index("Data")
        df_bench = pd.DataFrame({
            "Data": common_index, "Ultimo": closes[benchmark], "Apertura": opens[benchmark],
            "Massimo": highs[benchmark], "Minimo": lows[benchmark], "Volume": vols[benchmark]
        }).set_index("Data")
        return df_asset, df_bench
    except: return None, None

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

    y, x = df_asset["Var %"], df_bench["Var %"]
    covariance = np.cov(y, x)[0][1]
    variance = np.var(x, ddof=1) 
    if variance == 0: beta = 0 # Evita div by zero
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
            # Cerchiamo il nome pulito. Se non c'è in wiki, cerchiamo di indovinarlo o usiamo il ticker
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
        st.error("⚠️ La lista titoli è vuota. Cerca e aggiungi titoli nella sidebar.")
    else:
        with st.spinner(f'Analisi in corso...'):
            results = {}
            for t in final_selection:
                df_asset, df_bench = get_data_pair(t, benchmark_ticker, years_input)
                if df_asset is not None and not df_asset.empty:
                    df_a, df_b, stats = calculate_metrics_and_structure(df_asset, df_bench, rf_input, mrp_input)
                    results[t] = {"df_asset": df_a, "df_bench": df_b, "stats": stats}
            
            if results:
                st.session_state['analysis_results'] = results
                st.session_state['bench_used'] = selected_bench_name 
                st.session_state['done'] = True
                st.toast(f'✅ Completato!', icon="🚀")
            else:
                st.error("Nessun dato trovato. Controlla i ticker.")

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
        st.info(f"Analisi basata su **{years_input} anni** di dati settimanali.")

    excel_file = generate_excel_report(results, rf_input, mrp_input, bench_name)
    st.download_button("📥 Scarica Report Excel", data=excel_file, file_name="Analisi_Finanziaria_Pro.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")

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
