import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import plotly.express as px
import io 

# =========================
# CONFIGURAZIONE PAGINA
# =========================
st.set_page_config(page_title="CAPM Analysis Europe", layout="wide")

st.title("📊 CAPM Analysis: European Indices")
st.markdown("""
Questa applicazione calcola il **Costo del Capitale (CAPM)** e confronta i rendimenti attesi con quelli effettivi 
per i principali indici europei.
* **Metodologia:** Dati storici (Yahoo Finance) + Regressione OLS.
* **Parametri:** Basati su Survey Fernandez 2025 (modificabili).
""")

# =========================
# SIDEBAR - PARAMETRI UTENTE
# =========================
st.sidebar.header("⚙️ Parametri Modello")

rf_input = st.sidebar.number_input("Risk Free Rate (Bund 10Y)", value=2.5, step=0.1) / 100
mrp_input = st.sidebar.number_input("Market Risk Premium (Fernandez)", value=5.8, step=0.1) / 100
years_input = st.sidebar.slider("Orizzonte Temporale (Anni)", 3, 10, 5)

st.sidebar.markdown("---")
st.sidebar.info("Clicca su **'Avvia Analisi'** per aggiornare i dati.")

# =========================
# FUNZIONI DI CALCOLO
# =========================
@st.cache_data
def get_data(years):
    """Scarica i dati da Yahoo Finance"""
    tickers = ["^GDAXI", "^FCHI", "^IBEX", "FTSEMIB.MI", "^STOXX50E"]
    try:
        data = yf.download(tickers, period=f"{years}y", interval="1mo", auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            try:
                close = data["Close"]
            except KeyError:
                close = data.iloc[:, data.columns.get_level_values(0) == 'Close']
                close.columns = close.columns.droplevel(0)
        else:
            close = data["Close"]
        return close.dropna()
    except Exception as e:
        st.error(f"Errore download dati: {e}")
        return pd.DataFrame()

def calculate_capm(close_data, rf, mrp, years):
    """Esegue i calcoli CAPM e Regressione"""
    log_returns = np.log(close_data / close_data.shift(1)).dropna()
    market_index = "^STOXX50E"
    
    market_returns = log_returns[market_index]
    asset_returns = log_returns.drop(columns=market_index)
    
    results_list = []
    
    for asset in asset_returns.columns:
        Y = asset_returns[asset]
        X = sm.add_constant(market_returns)
        model = sm.OLS(Y, X).fit()
        
        beta = model.params.iloc[1]
        r_sq = model.rsquared
        p_value = model.pvalues.iloc[1]
        
        if beta < 0.8: natura = "Riduttivo (Strong)"
        elif beta < 1: natura = "Riduttivo (Mod.)"
        elif beta < 1.2: natura = "Amplificativo (Mod.)"
        else: natura = "Amplificativo (Aggr.)"
        
        expected_return = rf + beta * mrp
        start_p = close_data[asset].iloc[0]
        end_p = close_data[asset].iloc[-1]
        realized_return = (end_p / start_p) ** (1 / years) - 1
        deviation = realized_return - expected_return
        
        results_list.append({
            "Ticker": asset,
            "Beta": round(beta, 3),
            "Natura Rischio": natura,
            "R²": round(r_sq, 3),
            "P-Value": round(p_value, 4),
            "Exp. Return (CAPM)": round(expected_return * 100, 2),
            "Realized Return": round(realized_return * 100, 2),
            "Delta (Alpha)": round(deviation * 100, 2)
        })
        
    return pd.DataFrame(results_list), log_returns

def generate_excel(df_results, close_data, log_returns, rf, mrp):
    """Genera il file Excel in memoria"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name="Analisi CAPM", index=False)
        close_data.to_excel(writer, sheet_name="Prezzi Storici")
        log_returns.to_excel(writer, sheet_name="Rendimenti Log")
        
        params_df = pd.DataFrame({
            "Parametro": ["Risk Free Rate", "Market Risk Premium", "Benchmark"],
            "Valore": [f"{rf*100}%", f"{mrp*100}%", "STOXX Europe 50"]
        })
        params_df.to_excel(writer, sheet_name="Parametri", index=False)
        
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
                worksheet.column_dimensions[column_letter].width = max_length + 3
                
    return output.getvalue()

# =========================
# MAIN APP LOGIC (Gestione Stato)
# =========================

# 1. Se clicco il bottone, faccio i calcoli e SALVO nello stato
if st.button("🚀 Avvia Analisi", type="primary"):
    with st.spinner('Elaborazione in corso...'):
        close_df = get_data(years_input)
        if not close_df.empty:
            df_results, log_returns = calculate_capm(close_df, rf_input, mrp_input, years_input)
            
            # Salvo tutto nella "memoria" della sessione
            st.session_state['df_results'] = df_results
            st.session_state['close_df'] = close_df
            st.session_state['log_returns'] = log_returns
            st.session_state['analysis_done'] = True

# 2. Controllo se ho i dati in memoria (così rimangono anche dopo il refresh del download)
if st.session_state.get('analysis_done'):
    
    # Recupero i dati dalla memoria
    df_results = st.session_state['df_results']
    close_df = st.session_state['close_df']
    log_returns = st.session_state['log_returns']

    # --- MOSTRA RISULTATI ---
    st.subheader("📋 Risultati Analisi")
    st.dataframe(df_results.style.format({
        "Beta": "{:.3f}",
        "R²": "{:.3f}",
        "P-Value": "{:.4f}",
        "Exp. Return (CAPM)": "{:.2f}%",
        "Realized Return": "{:.2f}%",
        "Delta (Alpha)": "{:.2f}%"
    }), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Esposizione al Rischio (Beta)")
        fig_beta = px.bar(df_results, x="Ticker", y="Beta", color="Natura Rischio",
                          title="Beta vs Mercato (1.0)", text_auto=True)
        fig_beta.add_hline(y=1, line_dash="dash", annotation_text="Mercato")
        st.plotly_chart(fig_beta, use_container_width=True)
        
    with col2:
        st.subheader("Confronto Rendimenti")
        df_melted = df_results.melt(id_vars="Ticker", 
                                    value_vars=["Exp. Return (CAPM)", "Realized Return"],
                                    var_name="Tipo", value_name="Valore %")
        fig_returns = px.bar(df_melted, x="Ticker", y="Valore %", color="Tipo", barmode="group",
                             title="Atteso vs Reale", text_auto=True)
        st.plotly_chart(fig_returns, use_container_width=True)

    # --- DOWNLOAD BUTTON ---
    # Genero il file Excel usando i dati attuali
    excel_data = generate_excel(df_results, close_df, log_returns, rf_input, mrp_input)
    
    st.success("Analisi completata!")
    st.download_button(
        label="📥 Scarica Report Excel (.xlsx)",
        data=excel_data,
        file_name="CAPM_Analysis_Professional.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )
