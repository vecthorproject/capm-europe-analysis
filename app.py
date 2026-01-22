import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import io 

# =========================
# CONFIGURAZIONE PAGINA
# =========================
st.set_page_config(page_title="Analisi Beta & CAPM (Metodo Prof)", layout="wide")

st.title("📊 Analisi Beta & CAPM (Metodo Classico)")
st.markdown("""
Replica esatta della metodologia basata su:
* **Dati Completi:** Apertura, Massimo, Minimo, Chiusura, Volume.
* **Calcolo:** Variazione Percentuale Semplice (Var %).
* **Formula Beta:** Rapporto tra Covarianza e Varianza.
""")

# =========================
# SIDEBAR - PARAMETRI
# =========================
st.sidebar.header("⚙️ Input Dati")

default_tickers = "ENEL.MI, ISP.MI, ENI.MI"
user_tickers = st.sidebar.text_area("Inserisci Ticker (es. ENEL.MI)", default_tickers, height=100)
benchmark_ticker = "FTSEMIB.MI"

st.sidebar.markdown("---")
st.sidebar.header("Parametri CAPM")
rf_input = st.sidebar.number_input("Risk Free Rate (BTP 10Y)", value=3.8, step=0.1) / 100
mrp_input = st.sidebar.number_input("Market Risk Premium", value=5.5, step=0.1) / 100
years_input = st.sidebar.slider("Orizzonte Temporale (Anni)", 1, 5, 2) 

st.sidebar.info("Clicca **Avvia Analisi** per generare il report.")

# =========================
# FUNZIONI (MOTORE PROF)
# =========================

@st.cache_data
def get_detailed_data(ticker_string, years):
    """Scarica dati OHLCV completi per replica tabella prof"""
    tickers = [t.strip() for t in ticker_string.split(',') if t.strip() != ""]
    if benchmark_ticker not in tickers:
        tickers.append(benchmark_ticker)
    
    try:
        # Scarichiamo tutto (Open, High, Low, Close, Volume)
        data = yf.download(tickers, period=f"{years}y", interval="1wk", auto_adjust=False, progress=False)
        return data
    except Exception as e:
        st.error(f"Errore download: {e}")
        return pd.DataFrame()

def calculate_prof_metrics(data, rf, mrp, years):
    """Calcola Beta usando Covarianza/Varianza e Var % Semplice"""
    
    # 1. Estrazione e Pulizia Prezzi di Chiusura (Adj Close per i calcoli reali, Close per tabella)
    # Nota: Per replicare la prof usiamo 'Close' o 'Adj Close'. Solitamente Yahoo da Adj Close di default.
    # Usiamo 'Close' puro se vogliamo replicare i prezzi che vede a video, ma 'Adj Close' è finanziariamente corretto.
    # Per coerenza con la tabella "DataUltimo", usiamo Close.
    
    try:
        # Gestione colonne MultiIndex di yfinance
        closes = data["Close"] 
        opens = data["Open"]
        highs = data["High"]
        lows = data["Low"]
        vols = data["Volume"]
    except KeyError:
        st.error("Errore nella struttura dei dati scaricati.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 2. Calcolo Variazione Percentuale (Var %) - Metodo Semplice
    # Formula: (Prezzo_t - Prezzo_t-1) / Prezzo_t-1
    returns = closes.pct_change().dropna()
    
    if benchmark_ticker not in returns.columns:
        st.error("Benchmark mancante.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    market_ret = returns[benchmark_ticker]
    
    results_list = []
    
    # Creiamo anche un DataFrame gigante per l'Excel dettagliato
    detailed_export = pd.DataFrame()

    for asset in returns.columns:
        if asset == benchmark_ticker:
            continue
            
        asset_ret = returns[asset]
        
        # Allineamento dati (intersezione date)
        common_idx = asset_ret.index.intersection(market_ret.index)
        y = asset_ret.loc[common_idx]
        x = market_ret.loc[common_idx]
        
        # --- IL CUORE DEL CALCOLO (METODO PROF) ---
        # Covarianza (Asset, Mercato)
        covariance = np.cov(y, x)[0][1]
        # Varianza (Mercato)
        variance = np.var(x, ddof=1) # ddof=1 per varianza campionaria (Excel VAR.S)
        
        # Beta = Cov / Var
        beta = covariance / variance
        # ------------------------------------------
        
        # Classificazione
        if beta < 0.8: natura = "Riduttivo (Strong)"
        elif beta < 1: natura = "Riduttivo (Mod.)"
        elif beta < 1.2: natura = "Amplificativo (Mod.)"
        else: natura = "Amplificativo (Aggr.)"

        # CAPM & Performance
        expected_return = rf + beta * mrp
        
        start_p = closes[asset].iloc[0]
        end_p = closes[asset].iloc[-1]
        # CAGR Semplice
        realized_return = (end_p / start_p) ** (1 / years) - 1
        deviation = realized_return - expected_return
        
        results_list.append({
            "Ticker": asset,
            "Beta": round(beta, 3),
            "Covarianza": covariance, # Dato Prof
            "Varianza Mkt": variance, # Dato Prof
            "Natura": natura,
            "Rendimento Atteso (CAPM)": round(expected_return * 100, 2),
            "Rendimento Reale": round(realized_return * 100, 2),
            "Delta": round(deviation * 100, 2)
        })

        # --- PREPARAZIONE DATI PER EXCEL DETTAGLIATO (Replica Tabella) ---
        # Creiamo un blocco per questo ticker
        temp_df = pd.DataFrame({
            "Data": closes.index,
            "Ticker": asset,
            "Apertura": opens[asset],
            "Massimo": highs[asset],
            "Minimo": lows[asset],
            "Ultimo (Close)": closes[asset],
            "Volume": vols[asset],
            "Var %": returns[asset] * 100 # In formato percentuale leggibile
        }).sort_index(ascending=False) # Dal più recente al più vecchio come la prof
        
        detailed_export = pd.concat([detailed_export, temp_df])

    return pd.DataFrame(results_list), returns, detailed_export

def generate_prof_excel(summary_df, detailed_df, rf, mrp):
    """Genera Excel formattato come quello della prof"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        
        # Foglio 1: Sintesi (I Risultati)
        summary_df.to_excel(writer, sheet_name="Sintesi Beta CAPM", index=False)
        
        # Foglio 2: Dettaglio Storico (La tabella gigante)
        # Formattiamo le colonne per renderle leggibili
        detailed_df.to_excel(writer, sheet_name="Dati Storici Completi", index=False)
        
        # Foglio 3: Parametri
        params = pd.DataFrame({"Parametro": ["Risk Free", "MRP", "Benchmark"], "Valore": [rf, mrp, benchmark_ticker]})
        params.to_excel(writer, sheet_name="Parametri", index=False)
        
        # Formattazione colonne
        for sheet in writer.sheets:
            ws = writer.sheets[sheet]
            for col in ws.columns:
                try:
                    ws.column_dimensions[col[0].column_letter].width = 15
                except: pass

    return output.getvalue()

# =========================
# LOGICA APP
# =========================
if st.button("🚀 Avvia Analisi", type="primary"):
    with st.spinner('Scaricamento dati OHLCV e calcolo Covarianza...'):
        raw_data = get_detailed_data(user_tickers, years_input)
        
        if not raw_data.empty:
            df_results, returns_df, detailed_df = calculate_prof_metrics(raw_data, rf_input, mrp_input, years_input)
            
            st.session_state['results'] = df_results
            st.session_state['detailed'] = detailed_df
            st.session_state['done'] = True

if st.session_state.get('done'):
    df_results = st.session_state['results']
    detailed_df = st.session_state['detailed']
    
    st.subheader("📋 Risultati (Calcolo: Covarianza / Varianza)")
    st.dataframe(df_results.style.format({
        "Beta": "{:.4f}", 
        "Covarianza": "{:.6f}", 
        "Varianza Mkt": "{:.6f}",
        "Rendimento Atteso (CAPM)": "{:.2f}%", 
        "Rendimento Reale": "{:.2f}%"
    }), use_container_width=True)

    # Grafico veloce
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(df_results, x="Ticker", y="Beta", color="Natura", title="Beta", text_auto=True)
        fig.add_hline(y=1, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Esempio Dati Scaricati (Simile Prof)")
        # Mostriamo un'anteprima della tabella dettagliata (solo prime righe)
        st.dataframe(detailed_df.head(10).style.format({
            "Ultimo (Close)": "{:.2f}",
            "Apertura": "{:.2f}",
            "Var %": "{:.2f}%"
        }), use_container_width=True)

    # Download
    excel_file = generate_prof_excel(df_results, detailed_df, rf_input, mrp_input)
    st.download_button(
        label="📥 Scarica Excel (Formato Prof)",
        data=excel_file,
        file_name="Analisi_CAPM_Completa.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )

# =========================
# DOCUMENTAZIONE
# =========================
st.markdown("---")
with st.expander("📖 Nota Metodologica (Confronto con metodo Prof)"):
    st.markdown(r"""
    ### 1. Replica Esatta dei Dati
    Il sistema scarica i dati settimanali includendo:
    * **Prezzo Ultimo (Close):** Usato per il calcolo della variazione.
    * **Apertura, Massimo, Minimo, Volumi:** Inclusi nel file Excel per completezza visiva (come da tabella di riferimento).

    ### 2. Formula del Beta (Approccio Varianza/Covarianza)
    Invece della regressione OLS, qui replichiamo il calcolo manuale di Excel:
    $$ \beta = \frac{Cov(R_{asset}, R_{market})}{Var(R_{market})} $$
    
    Dove:
    * **Cov:** Covarianza tra i rendimenti del titolo e quelli del FTSE MIB.
    * **Var:** Varianza dei rendimenti del FTSE MIB.
    
    ### 3. Rendimenti (Var %)
    Viene utilizzata la variazione percentuale semplice, non logaritmica, per combaciare con la colonna "Var %" tipica dei fogli di calcolo finanziari di base:
    $$ Var\% = \frac{P_t - P_{t-1}}{P_{t-1}} $$
    """)
