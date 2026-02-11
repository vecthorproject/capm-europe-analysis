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
st.set_page_config(page_title="Financial Analytics Pro", layout="wide")

st.title("📊 Analisi Finanziaria Integrata: Prezzi & Bilanci")
st.markdown("""
Questa piattaforma combina l'analisi statistica del **Beta** con i dati fondamentali di bilancio per scorporare il rischio finanziario dal rischio operativo.
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
# MAPPATURA SETTORIALE (SIC -> ATECO)
# =========================
def map_industry_to_ateco(industry):
    """Mappatura euristica dei settori Yahoo Finance in codici ATECO 2007/NACE Rev. 2"""
    mapping = {
        "Banks—Diversified": "64.19 (Attività bancaria)",
        "Utilities—Renewable": "35.11 (Produzione energia elettrica)",
        "Auto Manufacturers": "29.10 (Fabbricazione autoveicoli)",
        "Oil & Gas Integrated": "06.10 (Estrazione petrolio/gas)",
        "Luxury Goods": "47.71 (Commercio abbigliamento/lusso)",
        "Insurance—Diversified": "65.12 (Assicurazioni)",
        "Telecommunications Services": "61.10 (Telecomunicazioni fisse)",
        "Pharmaceutical Retailers": "47.73 (Farmacie)"
    }
    return mapping.get(industry, "N.D. (Codifica non disponibile)")

# =========================
# 1. RICERCA E ANALISI FONDAMENTALE
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
                name = item.get('longname') or item.get('shortname')
                exchange = item.get('exchange') 
                if symbol and name:
                    results.append((f"{name} ({symbol}) - {exchange}", symbol, name))
        return results
    except: return []

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
# 2. MOTORE DI CALCOLO BETA (REGRESSIVO & HAMADA)
# =========================
def get_financial_metrics(ticker_obj):
    """Estrae i dati necessari per il Beta Unlevered"""
    try:
        info = ticker_obj.info
        bs = ticker_obj.balance_sheet
        
        market_cap = info.get('marketCap', 0)
        
        # Estrazione debito totale (gestione nomi righe variabili)
        total_debt = 0
        if not bs.empty:
            if 'Total Debt' in bs.index: total_debt = bs.loc['Total Debt'].iloc[0]
            elif 'Long Term Debt' in bs.index:
                lt_debt = bs.loc['Long Term Debt'].iloc[0]
                st_debt = bs.loc['Current Debt'].iloc[0] if 'Current Debt' in bs.index else 0
                total_debt = lt_debt + st_debt

        cash = bs.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in bs.index else 0
        net_debt = total_debt - cash
        
        return {
            "Market Cap": market_cap,
            "Total Debt": total_debt,
            "Net Debt": net_debt,
            "Industry": info.get('industry', 'N/A'),
            "Sector": info.get('sector', 'N/A')
        }
    except:
        return None

def calculate_unlevered_beta(levered_beta, debt, equity, tax_rate=0.24):
    """Formula di Hamada per l'Unlevered Beta"""
    if equity <= 0: return levered_beta
    return levered_beta / (1 + (1 - tax_rate) * (debt / equity))

# =========================
# 3. GESTIONE DATI STORICI (PREZZI)
# =========================
def get_historical_data(ticker, start, end, interval="1wk"):
    try:
        df = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
        if df.empty: return None
        
        # Standardizzazione colonne
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).normalize().tz_localize(None)
        
        # Resampling
        rule = "W-FRI" if interval == "1wk" else "ME"
        df_res = df.resample(rule).agg({
            "Close": "last", "Open": "first", "High": "max", "Low": "min", "Volume": "sum"
        }).dropna()
        
        df_res["Var %"] = df_res["Close"].pct_change()
        return df_res
    except: return None

# =========================
# SIDEBAR & INPUT
# =========================
st.sidebar.header("⚙️ Configurazione")
search_query = st.sidebar.text_input("Cerca Titolo (es. Ferrari, Eni):", "")

if search_query:
    search_results = search_yahoo_finance(search_query)
    if search_results:
        st.sidebar.selectbox("Risultati:", options=search_results, format_func=lambda x: x[0], key='temp_search_selection')
        st.sidebar.button("➕ Aggiungi", on_click=add_ticker_to_portfolio, type="primary")

st.sidebar.markdown("---")
bench_dict = {"FTSEMIB.MI": "Italia", "^STOXX50E": "Europa", "^GSPC": "USA"}
selected_bench = st.sidebar.selectbox("Benchmark:", list(bench_dict.keys()), format_func=lambda x: bench_dict[x])

rf_input = st.sidebar.number_input("Risk Free (%)", value=3.8) / 100
mrp_input = st.sidebar.number_input("Market Risk Premium (%)", value=5.5) / 100

# =========================
# LOGICA DI CALCOLO GENERALE
# =========================
if st.button("🚀 Avvia Analisi Integrata", type="primary"):
    if not st.session_state['selected_tickers_list']:
        st.error("Seleziona almeno un titolo.")
    else:
        full_results = {}
        with st.spinner("Estrazione dati di mercato e bilancio in corso..."):
            df_bench = get_historical_data(selected_bench, "2020-01-01", datetime.date.today().isoformat())
            
            for t in st.session_state['selected_tickers_list']:
                obj = yf.Ticker(t)
                df_asset = get_historical_data(t, "2020-01-01", datetime.date.today().isoformat())
                fundamentals = get_financial_metrics(obj)
                
                if df_asset is not None and df_bench is not None:
                    # Beta Regressivo
                    common = df_asset.index.intersection(df_bench.index)
                    y = df_asset.loc[common, "Var %"].dropna()
                    x = df_bench.loc[common, "Var %"].dropna()
                    common_clean = y.index.intersection(x.index)
                    
                    cov = np.cov(y.loc[common_clean], x.loc[common_clean])[0][1]
                    var_m = np.var(x.loc[common_clean])
                    b_levered = cov / var_m
                    
                    # Beta Unlevered (Hamada)
                    b_unlevered = b_levered
                    if fundamentals:
                        b_unlevered = calculate_unlevered_beta(
                            b_levered, fundamentals['Total Debt'], fundamentals['Market Cap']
                        )
                    
                    full_results[t] = {
                        "df_asset": df_asset.loc[common_clean],
                        "df_bench": df_bench.loc[common_clean],
                        "metrics": {
                            "Beta Levered (Market)": b_levered,
                            "Beta Unlevered (Asset)": b_unlevered,
                            "ATECO/NACE": map_industry_to_ateco(fundamentals['Industry']) if fundamentals else "N/A",
                            **fundamentals
                        } if fundamentals else {},
                        "financials": {
                            "Income": obj.financials,
                            "Balance": obj.balance_sheet,
                            "Cashflow": obj.cashflow
                        }
                    }

        st.session_state['analysis_done'] = full_results

# =========================
# VISUALIZZAZIONE & EXCEL
# =========================
if st.session_state.get('analysis_done'):
    results = st.session_state['analysis_done']
    
    # Sintesi
    st.subheader("📊 Confronto Rischio Operativo vs Finanziario")
    summary_df = pd.DataFrame([{
        "Società": st.session_state['ticker_names_map'][k],
        "Beta Levered": v['metrics'].get('Beta Levered (Market)'),
        "Beta Unlevered": v['metrics'].get('Beta Unlevered (Asset)'),
        "ATECO": v['metrics'].get('ATECO/NACE'),
        "D/E Ratio": v['metrics'].get('Total Debt', 0) / v['metrics'].get('Market Cap', 1)
    } for k, v in results.items()])
    st.dataframe(summary_df, use_container_width=True)

    # Download Excel
    def export_excel(res_dict):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Foglio Sintesi
            summary_df.to_excel(writer, sheet_name="Sintesi_Rischio", index=False)
            
            for ticker, data in res_dict.items():
                name = ticker.replace(".MI", "")
                # Foglio Dati Storici
                hist_df = data['df_asset'].copy()
                hist_df.to_excel(writer, sheet_name=f"{name}_Prezzi")
                
                # Foglio Bilancio
                if not data['financials']['Balance'].empty:
                    data['financials']['Balance'].to_excel(writer, sheet_name=f"{name}_Bilancio")
            
            # Formattazione centrata per tutti i fogli
            for sheet in writer.sheets.values():
                for row in sheet.iter_rows():
                    for cell in row:
                        cell.alignment = Alignment(horizontal='center')

        return output.getvalue()

    st.download_button("📥 Scarica Report Multi-Asset (Financials + Beta)", 
                       data=export_excel(results), 
                       file_name="Analisi_Completa_Beta_Financials.xlsx",
                       type="primary")
    
