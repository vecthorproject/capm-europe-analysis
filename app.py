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
    
# =========================
# SEZIONE DOCUMENTAZIONE (DA AGGIUNGERE ALLA FINE DEL FILE)
# =========================
st.markdown("---")
st.header("📚 Documentazione del Progetto")

with st.expander("📖 Leggi la Relazione Completa (Scopo, Metodologia e Codice)"):
    st.markdown("""
    ### 1. Scopo del Progetto
    L'obiettivo di questo applicativo è condurre una **verifica empirica ex-post** della teoria finanziaria moderna, mettendo a confronto il modello teorico (CAPM) con la realtà dei mercati azionari europei.
    
    In particolare, il progetto risponde a tre domande fondamentali:
    1.  **Quanto sono rischiosi** i singoli mercati nazionali (Italia, Germania, Francia, Spagna) rispetto alla media europea?
    2.  **Quale rendimento avrebbero dovuto offrire** teoricamente per compensare tale rischio?
    3.  **Hanno effettivamente pagato** quel rendimento negli ultimi 5 anni?
    
    ---

    ### 2. Flusso di Lavoro (Pipeline)
    L'analisi segue un processo strutturato in 4 fasi automatiche:
    
    1.  **Data Ingestion:** Il sistema scarica in tempo reale i prezzi di chiusura *adjusted* (rettificati per dividendi e split) da Yahoo Finance per gli ultimi 5 anni.
    2.  **Risk Assessment (Beta):** Viene calcolata la sensibilità di ogni indice nazionale rispetto al benchmark europeo (`^STOXX50E`) tramite regressione lineare.
    3.  **Pricing (CAPM):** Utilizzando parametri accademici (Risk Free e Market Premium), si calcola il "Prezzo del Rischio", ovvero il rendimento minimo atteso.
    4.  **Performance Evaluation:** Si confronta l'attesa con la realtà (CAGR effettivo) per determinare l'Alpha (extra-rendimento) o la sottoperformance.

    ---

    ### 3. Razionale Metodologico (Scelta dei Parametri)
    Per garantire rigore scientifico, i parametri di input non sono arbitrari ma derivano da standard accademici:
    
    * **Risk-Free Rate ($R_f$):** Viene utilizzato il rendimento del **Bund Tedesco a 10 anni** (approssimato al 2.5%). Rappresenta l'investimento privo di rischio nell'Eurozona, evitando le distorsioni legate allo spread dei titoli periferici (es. BTP).
    * **Market Risk Premium ($R_m - R_f$):** Il premio per il rischio è fissato al **5.8%**, basandosi sulla **Survey 2025 di Pablo Fernandez (IESE Business School)**, la fonte più autorevole per il consensus di analisti e accademici.
    * **Orizzonte Temporale:** 5 anni con dati mensili ($N=60$). La frequenza mensile è preferita a quella giornaliera per eliminare il "rumore" statistico di breve termine e stabilizzare la stima del Beta.

    ---
    
    ### 4. Spiegazione del Codice e Logica Matematica
    Di seguito viene analizzato il funzionamento del "motore" Python sottostante.

    #### A. Calcolo dei Rendimenti Logaritmici
    """)
    
    st.code("""
log_returns = np.log(close_data / close_data.shift(1)).dropna()
    """, language='python')
    
    st.markdown("""
    **Perché:** In finanza quantitativa si usano i rendimenti logaritmici (o *log-returns*) invece di quelli semplici perché sono additivi nel tempo e seguono meglio una distribuzione normale, requisito fondamentale per la regressione lineare.
    """)

    st.markdown("#### B. Calcolo del Beta (Regressione Lineare)")
    st.code("""
# Y = Asset (es. FTSEMIB), X = Mercato (STOXX50)
model = sm.OLS(Y, X).fit()
beta = model.params.iloc[1]
    """, language='python')
    
    st.markdown("""
    **Cosa succede:** Utilizziamo la libreria `statsmodels` per eseguire una regressione **OLS (Ordinary Least Squares)**. 
    Il **Beta ($\beta$)** rappresenta la pendenza della retta di regressione:
    * $\beta > 1$: L'indice è **Amplificativo** (Aggressivo).
    * $\beta < 1$: L'indice è **Riduttivo** (Difensivo).
    """)

    st.markdown("#### C. Formula del CAPM (Rendimento Atteso)")
    st.latex(r'''
    E(R_i) = R_f + \beta_i \times (R_m - R_f)
    ''')
    st.markdown("""
    Il codice applica questa formula per stabilire il rendimento teorico "giusto" dato il livello di rischio.
    """)

    st.markdown("#### D. Confronto con la Realtà (Rendimento Effettivo)")
    st.code("""
realized_return = (end_p / start_p) ** (1 / years) - 1
deviation = realized_return - expected_return
    """, language='python')
    
    st.markdown("""
    **Il Delta (Alpha Proxy):** Infine, calcoliamo il rendimento reale composto annualizzato (**CAGR**) e lo sottraiamo al rendimento atteso CAPM.
    * **Delta Positivo:** Il mercato ha sovraperformato le attese teoriche.
    * **Delta Negativo:** Il rischio assunto non è stato remunerato adeguatamente dal mercato.
    """)
    
    st.info("Questa architettura garantisce che i risultati siano basati su dati reali e metodologie trasparenti, permettendo una verifica istantanea delle ipotesi di mercato.")
