import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import io 

# =========================
# CONFIGURAZIONE PAGINA
# =========================
st.set_page_config(page_title="Analisi Finanziaria Beta & CAPM", layout="wide")

st.title("📊 Analisi Finanziaria: Beta, CAPM e Dati Storici")
st.markdown("""
Strumento professionale per il calcolo del **Costo del Capitale** e del **Rischio Sistematico (Beta)**.
Confronto diretto (Side-by-Side) tra i titoli selezionati e l'indice di riferimento (Benchmark).
""")

# =========================
# DATABASE INDICI E TITOLI
# =========================

# Mappa dei principali indici mondiali (Nome Visualizzato -> Ticker Yahoo)
BENCHMARK_OPTIONS = {
    "FTSE MIB (Italia 40 - Principale)": "FTSEMIB.MI",
    "FTSE Italia All-Share (Tutto il listino)": "^FTITLMS",
    "DAX (Germania 40)": "^GDAXI",
    "CAC 40 (Francia)": "^FCHI",
    "S&P 500 (USA)": "^GSPC",
    "Euro Stoxx 50 (Europa)": "^STOXX50E"
}

# Lista titoli FTSE MIB precaricati
FTSE_MIB_TICKERS = [
    "A2A.MI", "AMP.MI", "AZM.MI", "BGN.MI", "BMED.MI", "BAMI.MI", "BPE.MI", 
    "CPR.MI", "DIA.MI", "ENEL.MI", "ENI.MI", "ERG.MI", "RACE.MI", "FBK.MI", 
    "G.MI", "HER.MI", "IP.MI", "ISP.MI", "INW.MI", "IG.MI", "IVG.MI", "LDO.MI", 
    "MB.MI", "MONC.MI", "NEXI.MI", "PIRC.MI", "PST.MI", "PRY.MI", "REC.MI", 
    "SPM.MI", "SRG.MI", "STLAM.MI", "STMMI.MI", "TEN.MI", "TRN.MI", "TIT.MI", 
    "UCG.MI", "UNI.MI"
]

# =========================
# SIDEBAR - INPUT
# =========================
st.sidebar.header("⚙️ Configurazione Analisi")

# 1. Selezione Titoli (Asset)
st.sidebar.subheader("1. Selezione Asset")
selected_tickers = st.sidebar.multiselect(
    "Scegli i titoli:",
    options=FTSE_MIB_TICKERS,
    default=["ENEL.MI"], 
    help="Digita per cercare o seleziona dalla lista."
)

st.sidebar.markdown("---")

# 2. Selezione Benchmark (Indice)
st.sidebar.subheader("2. Selezione Benchmark")
selected_bench_name = st.sidebar.selectbox(
    "Confronta con Indice:",
    options=list(BENCHMARK_OPTIONS.keys()),
    index=0 # Default: FTSE MIB
)
# Recuperiamo il ticker vero (es. FTSEMIB.MI) dal nome scelto
benchmark_ticker = BENCHMARK_OPTIONS[selected_bench_name]

st.sidebar.markdown("---")

# 3. Parametri CAPM
st.sidebar.subheader("3. Parametri Macro")
rf_input = st.sidebar.number_input("Risk Free Rate (BTP 10Y)", value=3.8, step=0.1) / 100
mrp_input = st.sidebar.number_input("Market Risk Premium", value=5.5, step=0.1) / 100
years_input = st.sidebar.slider("Orizzonte Temporale (Anni)", 1, 5, 2) 

st.sidebar.info("Modifica i parametri e premi **Avvia Analisi**.")

# =========================
# MOTORE DI CALCOLO
# =========================

@st.cache_data
def get_data_pair(ticker, benchmark, years):
    """Scarica i dati per una coppia Ticker-Benchmark e li allinea"""
    try:
        data = yf.download([ticker, benchmark], period=f"{years}y", interval="1wk", auto_adjust=False, progress=False)
        
        try:
            closes = data["Close"][[ticker, benchmark]].dropna()
            opens = data["Open"][[ticker, benchmark]].dropna()
            highs = data["High"][[ticker, benchmark]].dropna()
            lows = data["Low"][[ticker, benchmark]].dropna()
            vols = data["Volume"][[ticker, benchmark]].dropna()
        except KeyError:
            return None, None

        common_index = closes.index
        
        df_asset = pd.DataFrame({
            "Data": common_index,
            "Ultimo": closes[ticker],
            "Apertura": opens[ticker],
            "Massimo": highs[ticker],
            "Minimo": lows[ticker],
            "Volume": vols[ticker]
        }).set_index("Data")

        df_bench = pd.DataFrame({
            "Data": common_index,
            "Ultimo": closes[benchmark],
            "Apertura": opens[benchmark],
            "Massimo": highs[benchmark],
            "Minimo": lows[benchmark],
            "Volume": vols[benchmark]
        }).set_index("Data")
        
        return df_asset, df_bench

    except Exception as e:
        return None, None

def calculate_metrics_and_structure(df_asset, df_bench, rf, mrp, years):
    """Calcola Beta, Var% e prepara i dati separati"""
    
    # Calcolo Rendimenti
    df_asset["Var %"] = df_asset["Ultimo"].pct_change()
    df_asset["Rendimento %"] = df_asset["Var %"]
    
    df_bench["Var %"] = df_bench["Ultimo"].pct_change()
    df_bench["Rendimento %"] = df_bench["Var %"]

    df_asset = df_asset.dropna()
    df_bench = df_bench.dropna()
    
    common_idx = df_asset.index.intersection(df_bench.index)
    df_asset = df_asset.loc[common_idx].sort_index(ascending=False)
    df_bench = df_bench.loc[common_idx].sort_index(ascending=False)

    # Beta
    y = df_asset["Var %"]
    x = df_bench["Var %"]
    
    covariance = np.cov(y, x)[0][1]
    variance = np.var(x, ddof=1) 
    beta = covariance / variance
    
    expected_return = rf + beta * mrp
    
    stats = {
        "Beta": beta,
        "Covarianza": covariance,
        "Varianza": variance,
        "Exp Return": expected_return
    }
    
    return df_asset, df_bench, stats

def generate_excel_report(analysis_results, rf, mrp, bench_name):
    """Genera Excel con NOME INDICE DINAMICO"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        
        def fmt_pct(val):
            if pd.isna(val) or isinstance(val, str): return val
            return f"{val * 100:.3f}%"
        
        # Foglio Sintesi
        summary_data = []
        for ticker, data in analysis_results.items():
            summary_data.append({
                "Ticker": ticker,
                "Beta": f"{data['stats']['Beta']:.3f}",
                "Covarianza": f"{data['stats']['Covarianza']:.6f}",
                "Varianza Mkt": f"{data['stats']['Varianza']:.6f}",
                "Rendimento Atteso (CAPM)": fmt_pct(data['stats']['Exp Return'])
            })
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Sintesi", index=False)
        
        # Fogli Dettaglio
        for ticker, data in analysis_results.items():
            sheet_name = ticker.replace(".MI", "")[:30] 
            
            # Intestazione Metriche
            risk_free_fmt = fmt_pct(rf)
            mrp_fmt = fmt_pct(mrp)
            capm_fmt = fmt_pct(data['stats']['Exp Return'])
            
            metrics_df = pd.DataFrame({
                "METRICA": ["BETA", "COVARIANZA", "VARIANZA", "RISK FREE", "MRP", "CAPM RETURN"],
                "VALORE": [
                    f"{data['stats']['Beta']:.4f}",
                    f"{data['stats']['Covarianza']:.6f}",
                    f"{data['stats']['Varianza']:.6f}",
                    risk_free_fmt,
                    mrp_fmt,
                    capm_fmt
                ]
            })
            metrics_df.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0, index=False)
            
            # Dati Storici
            df_asset_view = data['df_asset'].copy()
            df_bench_view = data['df_bench'].copy()
            
            for col in ["Var %", "Rendimento %"]:
                if col in df_asset_view.columns:
                    df_asset_view[col] = df_asset_view[col].apply(fmt_pct)
                if col in df_bench_view.columns:
                    df_bench_view[col] = df_bench_view[col].apply(fmt_pct)

            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            
            # Scrittura Side-by-Side
            worksheet.cell(row=9, column=1, value=ticker) 
            df_asset_view.to_excel(writer, sheet_name=sheet_name, startrow=9, startcol=0)
            
            offset = len(df_asset_view.columns) + 1 + 2 
            # Qui usiamo il nome dinamico del benchmark scelto dall'utente
            worksheet.cell(row=9, column=offset + 1, value=f"{bench_name} (Benchmark)")
            df_bench_view.to_excel(writer, sheet_name=sheet_name, startrow=9, startcol=offset)
            
        # Auto-width
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length: max_length = len(str(cell.value))
                    except: pass
                adjusted_width = (max_length + 2)
                if adjusted_width < 12: adjusted_width = 12
                worksheet.column_dimensions[column_letter].width = adjusted_width
            
    return output.getvalue()

# =========================
# LOGICA APP
# =========================
if st.button("🚀 Avvia Analisi", type="primary"):
    
    tickers_list = selected_tickers
    
    if not tickers_list:
        st.error("⚠️ Seleziona almeno un titolo dalla lista.")
    else:
        with st.spinner(f'Confronto titoli vs {selected_bench_name}...'):
            results = {}
            
            for t in tickers_list:
                df_asset, df_bench = get_data_pair(t, benchmark_ticker, years_input)
                
                if df_asset is not None and not df_asset.empty:
                    df_a, df_b, stats = calculate_metrics_and_structure(df_asset, df_bench, rf_input, mrp_input, years_input)
                    results[t] = {
                        "df_asset": df_a,
                        "df_bench": df_b,
                        "stats": stats
                    }
            
            if results:
                st.session_state['analysis_results'] = results
                st.session_state['bench_used'] = selected_bench_name # Salviamo il nome del benchmark usato
                st.session_state['done'] = True
                st.toast(f'✅ Analisi completata con {selected_bench_name}!', icon="🚀")
            else:
                st.error("Impossibile scaricare i dati. Controlla la connessione.")

# =========================
# VISUALIZZAZIONE
# =========================
if st.session_state.get('done'):
    results = st.session_state['analysis_results']
    bench_name = st.session_state.get('bench_used', selected_bench_name)
    
    summary_list = []
    for t, data in results.items():
        summary_list.append({
            "Ticker": t,
            "Beta": data['stats']['Beta'],
            "CAPM Return": f"{data['stats']['Exp Return']*100:.2f}%"
        })
    
    st.subheader("📋 Sintesi Risultati")
    st.dataframe(pd.DataFrame(summary_list), use_container_width=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Confronto Rischio (Beta)")
        beta_df = pd.DataFrame(summary_list)
        fig = px.bar(beta_df, x="Ticker", y="Beta", text_auto=".2f", title=f"Beta vs {bench_name} (1.0)")
        fig.add_hline(y=1, line_dash="dash", annotation_text="Mercato")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.info(f"Analisi basata sul confronto con **{bench_name}**.")

    # Download Excel con parametro nome benchmark
    excel_file = generate_excel_report(results, rf_input, mrp_input, bench_name)
    st.download_button(
        label="📥 Scarica Report Excel",
        data=excel_file,
        file_name="Analisi_Finanziaria_Completa.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )

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
