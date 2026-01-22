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
Strumento di calcolo del **Costo del Capitale** e del **Rischio Sistematico (Beta)**.
Confronto diretto (Side-by-Side) tra Asset e Benchmark (FTSE MIB) su base settimanale.
""")

# =========================
# SIDEBAR - INPUT
# =========================
st.sidebar.header("⚙️ Configurazione Analisi")

default_tickers = "ENEL.MI, ISP.MI, ENI.MI"
user_tickers = st.sidebar.text_area("Ticker (es. ENEL.MI, ISP.MI)", default_tickers, height=100)
benchmark_ticker = "FTSEMIB.MI"

st.sidebar.markdown("---")
st.sidebar.header("Parametri CAPM")
rf_input = st.sidebar.number_input("Risk Free Rate (BTP 10Y)", value=3.8, step=0.1) / 100
mrp_input = st.sidebar.number_input("Market Risk Premium (Fernandez)", value=5.5, step=0.1) / 100
years_input = st.sidebar.slider("Orizzonte Temporale (Anni)", 1, 5, 2) 

st.sidebar.info("Premi **Avvia Analisi** per generare il report.")

# =========================
# MOTORE DI CALCOLO
# =========================

@st.cache_data
def get_data_pair(ticker, benchmark, years):
    """Scarica i dati per una coppia Ticker-Benchmark e li allinea"""
    try:
        # Scarichiamo entrambi i ticker
        data = yf.download([ticker, benchmark], period=f"{years}y", interval="1wk", auto_adjust=False, progress=False)
        
        # Estraiamo i dati grezzi
        try:
            closes = data["Close"][[ticker, benchmark]].dropna()
            opens = data["Open"][[ticker, benchmark]].dropna()
            highs = data["High"][[ticker, benchmark]].dropna()
            lows = data["Low"][[ticker, benchmark]].dropna()
            vols = data["Volume"][[ticker, benchmark]].dropna()
        except KeyError:
            return None, None

        # Allineamento date (Intersezione)
        common_index = closes.index
        
        # Creiamo DataFrame Asset
        df_asset = pd.DataFrame({
            "Data": common_index,
            "Ultimo": closes[ticker],
            "Apertura": opens[ticker],
            "Massimo": highs[ticker],
            "Minimo": lows[ticker],
            "Volume": vols[ticker]
        }).set_index("Data")

        # Creiamo DataFrame Benchmark
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
    
    # 1. Calcolo Variazione % (Rendimento Semplice)
    df_asset["Var %"] = df_asset["Ultimo"].pct_change()
    df_asset["Rendimento %"] = df_asset["Var %"]
    
    df_bench["Var %"] = df_bench["Ultimo"].pct_change()
    df_bench["Rendimento %"] = df_bench["Var %"]

    # Rimuoviamo la prima riga che è NaN
    df_asset = df_asset.dropna()
    df_bench = df_bench.dropna()
    
    # Ri-allineamento
    common_idx = df_asset.index.intersection(df_bench.index)
    df_asset = df_asset.loc[common_idx].sort_index(ascending=False) # Dal più recente
    df_bench = df_bench.loc[common_idx].sort_index(ascending=False)

    # 2. Calcolo Beta (Covarianza / Varianza)
    y = df_asset["Var %"]
    x = df_bench["Var %"]
    
    covariance = np.cov(y, x)[0][1]
    variance = np.var(x, ddof=1) 
    beta = covariance / variance
    
    # 3. Parametri CAPM
    expected_return = rf + beta * mrp
    
    stats = {
        "Beta": beta,
        "Covarianza": covariance,
        "Varianza": variance,
        "Exp Return": expected_return
    }
    
    # Restituiamo i due dataframe separati per gestirli meglio nell'Excel
    return df_asset, df_bench, stats

def generate_excel_report(analysis_results, rf, mrp):
    """Genera Excel con formattazione % e SPAZIATURA"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        
        # --- Helper per formattare in % ---
        def fmt_pct(val):
            """Converte 0.05123 in '5.123%'"""
            if pd.isna(val): return ""
            return f"{val * 100:.3f}%"
        
        # Foglio Riepilogo
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
        
        # Un Foglio per ogni Ticker
        for ticker, data in analysis_results.items():
            sheet_name = ticker.replace(".MI", "")[:30] 
            
            # --- 1. Preparazione Metriche in Alto ---
            # Applichiamo la formattazione % solo dove serve
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
            
            # --- 2. Preparazione Dati Storici (Formattazione Colonne) ---
            # Creiamo copie per non rovinare i dati originali in session_state
            df_asset_view = data['df_asset'].copy()
            df_bench_view = data['df_bench'].copy()
            
            # Formattiamo le colonne percentuali
            cols_to_format = ["Var %", "Rendimento %"]
            
            for col in cols_to_format:
                if col in df_asset_view.columns:
                    df_asset_view[col] = df_asset_view[col].apply(fmt_pct)
                if col in df_bench_view.columns:
                    df_bench_view[col] = df_bench_view[col].apply(fmt_pct)

            # Recuperiamo l'oggetto worksheet
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            
            # --- SCRITTURA BLOCCO SINISTRO (ASSET) ---
            worksheet.cell(row=9, column=1, value=ticker) 
            df_asset_view.to_excel(writer, sheet_name=sheet_name, startrow=9, startcol=0)
            
            # --- SCRITTURA BLOCCO DESTRO (MERCATO) ---
            offset = len(df_asset_view.columns) + 1 + 2 # +1 Index, +2 Spazio
            worksheet.cell(row=9, column=offset + 1, value="FTSE MIB (Benchmark)")
            df_bench_view.to_excel(writer, sheet_name=sheet_name, startrow=9, startcol=offset)
            
        # ==========================================
        # AUTO-ADJUST COLUMNS (Anti ####)
        # ==========================================
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                
                adjusted_width = (max_length + 2)
                if adjusted_width < 12: adjusted_width = 12
                worksheet.column_dimensions[column_letter].width = adjusted_width
            
    return output.getvalue()

# =========================
# LOGICA APP
# =========================
if st.button("🚀 Avvia Analisi", type="primary"):
    with st.spinner('Elaborazione Side-by-Side in corso...'):
        
        tickers = [t.strip() for t in user_tickers.split(',') if t.strip() != ""]
        results = {}
        
        for t in tickers:
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
            st.session_state['done'] = True
        else:
            st.error("Nessun dato trovato. Controlla i ticker.")

if st.session_state.get('done'):
    results = st.session_state['analysis_results']
    
    # Tabella Riepilogo a Video
    summary_list = []
    for t, data in results.items():
        summary_list.append({
            "Ticker": t,
            "Beta": data['stats']['Beta'],
            "CAPM Return": f"{data['stats']['Exp Return']*100:.2f}%"
        })
    
    st.subheader("📋 Sintesi Risultati")
    st.dataframe(pd.DataFrame(summary_list), use_container_width=True)
    
    # Grafico Beta
    st.subheader("Confronto Rischio (Beta)")
    beta_df = pd.DataFrame(summary_list)
    fig = px.bar(beta_df, x="Ticker", y="Beta", text_auto=".2f", title="Beta vs Mercato (1.0)")
    fig.add_hline(y=1, line_dash="dash", annotation_text="Mercato")
    st.plotly_chart(fig, use_container_width=True)

    # Download Excel
    excel_file = generate_excel_report(results, rf_input, mrp_input)
    st.download_button(
        label="📥 Scarica Report Excel (Formattato)",
        data=excel_file,
        file_name="Analisi_Finanziaria_Completa.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )

# =========================
# RELAZIONE METODOLOGICA
# =========================
st.markdown("---")
with st.expander("📚 Metodologia e Fonte Dati"):
    st.markdown(r"""
    ### 1. Struttura dei Dati
    L'analisi scarica e processa le serie storiche **settimanali** (Weekly timeframe) per garantire significatività statistica su orizzonti di breve/medio periodo (2-5 anni).
    
    Il file Excel generato presenta, per ogni titolo, una struttura **Side-by-Side** separata da due colonne vuote:
    * **Lato Sinistro:** Intestato col nome del Ticker. Contiene Dati OHLCV e Rendimenti.
    * **Lato Destro:** Intestato "FTSE MIB (Benchmark)". Contiene Dati OHLCV e Rendimenti del mercato.
    
    ### 2. Calcolo dei Parametri di Rischio
    Il coefficiente **Beta ($\beta$)** è calcolato esplicitamente attraverso il rapporto tra Covarianza e Varianza:
    
    $$ \beta = \frac{Cov(R_{asset}, R_{market})}{Var(R_{market})} $$
    
    * **$R_{asset}$:** Variazione percentuale settimanale del titolo.
    * **$R_{market}$:** Variazione percentuale settimanale del FTSE MIB.
    
    ### 3. Significato del CAPM Return
    Il valore indicato come "CAPM Return" (o Rendimento Atteso) indica il rendimento teorico annuo specifico per quel titolo, calcolato come:
    
    $$ E(R) = R_f + \beta \times (R_m - R_f) $$
    
    Esso risponde alla domanda: *"Quanto dovrebbe rendere questo titolo per compensare il rischio specifico (Beta) che sto assumendo rispetto a un BTP?"*.
    
    * **Risk-Free Rate ($R_f$):** BTP Italia 10 Anni (Default: ~3.8%). 
    * **Market Risk Premium ($MRP$):** 5.5% (Survey IESE Pablo Fernandez, 2025).
    """)
