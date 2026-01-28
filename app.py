import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import io 
import requests # Necessario per aggirare il blocco 403

# =========================
# CONFIGURAZIONE PAGINA
# =========================
st.set_page_config(page_title="Analisi Finanziaria Beta & CAPM", layout="wide")

st.title("📊 Analisi Finanziaria: Beta, CAPM e Dati Storici")
st.markdown("""
Strumento professionale per il calcolo del **Costo del Capitale** e del **Rischio Sistematico (Beta)**.
I titoli italiani vengono recuperati **dinamicamente** da Wikipedia, ma puoi aggiungere qualsiasi ticker globale manualmente.
""")

# =========================
# FUNZIONE RECUPERO DINAMICO (ANTI-BLOCCO 403)
# =========================
@st.cache_data
def get_ftse_mib_tickers_dynamic():
    """
    Tenta di scaricare da Wikipedia fingendosi un browser.
    Se fallisce, usa una lista statica di backup.
    """
    # Lista di riserva (Fallback) "Indistruttibile"
    STATIC_BACKUP = {
        "A2A.MI": "A2A", "AMP.MI": "Amplifon", "AZM.MI": "Azimut", "BGN.MI": "Banca Generali",
        "BMED.MI": "Banca Mediolanum", "BAMI.MI": "Banco BPM", "BPE.MI": "BPER Banca",
        "CPR.MI": "Campari", "DIA.MI": "Diasorin", "ENEL.MI": "Enel", "ENI.MI": "Eni",
        "ERG.MI": "ERG", "RACE.MI": "Ferrari", "FBK.MI": "FinecoBank", "G.MI": "Generali",
        "HER.MI": "Hera", "IP.MI": "Interpump", "ISP.MI": "Intesa Sanpaolo", "INW.MI": "Inwit",
        "IG.MI": "Italgas", "IVG.MI": "Iveco", "LDO.MI": "Leonardo", "MB.MI": "Mediobanca",
        "MONC.MI": "Moncler", "NEXI.MI": "Nexi", "PIRC.MI": "Pirelli", "PST.MI": "Poste Italiane",
        "PRY.MI": "Prysmian", "REC.MI": "Recordati", "SPM.MI": "Saipem", "SRG.MI": "Snam",
        "STLAM.MI": "Stellantis", "STMMI.MI": "STMicroelectronics", "TEN.MI": "Tenaris",
        "TRN.MI": "Terna", "TIT.MI": "Telecom Italia", "UCG.MI": "UniCredit", "UNI.MI": "Unipol"
    }

    try:
        url = "https://en.wikipedia.org/wiki/FTSE_MIB"
        
        # IL TRUCCO: Ci fingiamo un browser Chrome per evitare l'errore 403
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # Scarichiamo l'HTML con requests
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Controlla se ci sono errori
        
        # Passiamo l'HTML a Pandas
        tables = pd.read_html(io.StringIO(response.text))
        df_list = tables[1] 
        
        tickers_dict = {}
        for index, row in df_list.iterrows():
            try:
                # Adattiamo in base alla struttura attuale di Wiki
                ticker = row.iloc[1] 
                name = row.iloc[0]
                if "MI" not in ticker: ticker = f"{ticker}.MI"
                tickers_dict[ticker] = name
            except:
                continue
            
        if not tickers_dict: return STATIC_BACKUP # Se la tabella è vuota, usa backup
        return tickers_dict

    except Exception as e:
        # Se fallisce (no internet, wikipedia giù, ecc), usa il backup silenziosamente
        return STATIC_BACKUP

# Carichiamo la lista (O da Wikipedia, o dal Backup)
DYNAMIC_TICKER_MAP = get_ftse_mib_tickers_dynamic()

# =========================
# DATABASE INDICI
# =========================
BENCHMARK_OPTIONS = {
    "FTSE MIB (Italia 40)": "FTSEMIB.MI",
    "FTSE Italia All-Share": "^FTITLMS",
    "DAX (Germania)": "^GDAXI",
    "CAC 40 (Francia)": "^FCHI",
    "S&P 500 (USA)": "^GSPC",
    "Euro Stoxx 50 (Europa)": "^STOXX50E",
    "Nasdaq 100 (Tech USA)": "^NDX"
}

# =========================
# SIDEBAR - INPUT
# =========================
st.sidebar.header("⚙️ Configurazione Analisi")

st.sidebar.subheader("1. Selezione Asset")

# A. Menu a tendina (Popolato da Wikipedia o Backup)
def format_func_ticker(option):
    return f"{DYNAMIC_TICKER_MAP.get(option, option)} ({option})"

selected_from_list = []
if DYNAMIC_TICKER_MAP:
    selected_from_list = st.sidebar.multiselect(
        "A. Scegli dal FTSE MIB:",
        options=list(DYNAMIC_TICKER_MAP.keys()),
        default=["ENEL.MI", "ISP.MI"] if "ENEL.MI" in DYNAMIC_TICKER_MAP else [],
        format_func=format_func_ticker
    )

# B. Input Manuale
st.sidebar.markdown("**Oppure aggiungi ticker extra:**")
manual_tickers_str = st.sidebar.text_input(
    "B. Inserimento Manuale (es. RACE.MI, AAPL, TSLA)",
    help="Inserisci ticker separati da virgola."
)

manual_tickers_list = [t.strip() for t in manual_tickers_str.split(',') if t.strip() != ""]
final_tickers_list = list(set(selected_from_list + manual_tickers_list)) 

st.sidebar.markdown("---")

# 2. Selezione Benchmark
st.sidebar.subheader("2. Selezione Benchmark")
selected_bench_name = st.sidebar.selectbox(
    "Confronta con Indice:",
    options=list(BENCHMARK_OPTIONS.keys()),
    index=0 
)
benchmark_ticker = BENCHMARK_OPTIONS[selected_bench_name]

st.sidebar.markdown("---")

# 3. Parametri CAPM
st.sidebar.subheader("3. Parametri Macro")
rf_input = st.sidebar.number_input("Risk Free Rate (BTP 10Y)", value=3.8, step=0.1) / 100
mrp_input = st.sidebar.number_input("Market Risk Premium", value=5.5, step=0.1) / 100
years_input = st.sidebar.slider("Orizzonte Temporale (Anni)", 1, 5, 2) 

st.sidebar.info(f"Titoli selezionati: {len(final_tickers_list)}. Premi **Avvia Analisi**.")

# =========================
# MOTORE DI CALCOLO
# =========================

@st.cache_data
def get_data_pair(ticker, benchmark, years):
    try:
        data = yf.download([ticker, benchmark], period=f"{years}y", interval="1wk", auto_adjust=False, progress=False)
        try:
            closes = data["Close"][[ticker, benchmark]].dropna()
            opens = data["Open"][[ticker, benchmark]].dropna()
            highs = data["High"][[ticker, benchmark]].dropna()
            lows = data["Low"][[ticker, benchmark]].dropna()
            vols = data["Volume"][[ticker, benchmark]].dropna()
        except KeyError: return None, None

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
    except Exception: return None, None

def calculate_metrics_and_structure(df_asset, df_bench, rf, mrp, years):
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
    beta = covariance / variance
    expected_return = rf + beta * mrp
    
    stats = {"Beta": beta, "Covarianza": covariance, "Varianza": variance, "Exp Return": expected_return}
    return df_asset, df_bench, stats

def generate_excel_report(analysis_results, rf, mrp, bench_name):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        def fmt_pct(val): return f"{val * 100:.3f}%" if (not pd.isna(val) and not isinstance(val, str)) else val
        
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
            sheet_name = ticker.replace(".MI", "")[:30] 
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
                max_len = 0
                for cell in col:
                    try: 
                        if len(str(cell.value)) > max_len: max_len = len(str(cell.value))
                    except: pass
                ws.column_dimensions[col[0].column_letter].width = max(max_len + 2, 12)
            
    return output.getvalue()

# =========================
# LOGICA APP
# =========================
if st.button("🚀 Avvia Analisi", type="primary"):
    
    if not final_tickers_list:
        st.error("⚠️ Inserisci almeno un titolo (dalla lista o manualmente).")
    else:
        with st.spinner(f'Analisi in corso...'):
            results = {}
            for t in final_tickers_list:
                df_asset, df_bench = get_data_pair(t, benchmark_ticker, years_input)
                if df_asset is not None and not df_asset.empty:
                    df_a, df_b, stats = calculate_metrics_and_structure(df_asset, df_bench, rf_input, mrp_input, years_input)
                    results[t] = {"df_asset": df_a, "df_bench": df_b, "stats": stats}
            
            if results:
                st.session_state['analysis_results'] = results
                st.session_state['bench_used'] = selected_bench_name 
                st.session_state['done'] = True
                st.toast(f'✅ Completato per {len(results)} ticker!', icon="🚀")
            else:
                st.error("Nessun dato trovato. Controlla i ticker inseriti.")

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
        st.subheader("Confronto Rischio (Beta)")
        beta_df = pd.DataFrame(summary_list)
        fig = px.bar(beta_df, x="Società", y="Beta", text_auto=".2f", title=f"Beta vs {bench_name} (1.0)")
        fig.add_hline(y=1, line_dash="dash", annotation_text="Mercato")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.info(f"Benchmark utilizzato: **{bench_name}**.")

    excel_file = generate_excel_report(results, rf_input, mrp_input, bench_name)
    st.download_button("📥 Scarica Report Excel", data=excel_file, file_name="Analisi_Finanziaria_Completa.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")

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
