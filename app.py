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
    """Mappatura dei settori Yahoo Finance in codici ATECO 2007/NACE Rev. 2"""
    mapping = {
        "Banks—Diversified": "64.19 (Attività bancaria)",
        "Utilities—Renewable": "35.11 (Produzione energia elettrica)",
        "Auto Manufacturers": "29.10 (Fabbricazione autoveicoli)",
        "Oil & Gas Integrated": "06.10 (Estrazione petrolio/gas)",
        "Luxury Goods": "47.71 (Commercio abbigliamento/lusso)",
        "Insurance—Diversified": "65.12 (Assicurazioni)",
        "Telecommunications Services": "61.10 (Telecomunicazioni fisse)",
        "Pharmaceutical Retailers": "47.73 (Farmacie)",
        "Aerospace & Defense": "30.30 (Aerospaziale)",
        "Semiconductors": "26.11 (Semiconduttori)"
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
# 2. MOTORE DI CALCOLO BETA
# =========================
def get_financial_metrics(ticker_obj):
    try:
        info = ticker_obj.info
        bs = ticker_obj.balance_sheet
        market_cap = info.get('marketCap', 0)
        total_debt = 0
        if not bs.empty:
            if 'Total Debt' in bs.index: total_debt = bs.loc['Total Debt'].iloc[0]
            elif 'Long Term Debt' in bs.index:
                lt_debt = bs.loc['Long Term Debt'].iloc[0]
                st_debt = bs.loc['Current Debt'].iloc[0] if 'Current Debt' in bs.index else 0
                total_debt = lt_debt + st_debt
        cash = bs.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in bs.index else 0
        return {
            "Market Cap": market_cap,
            "Total Debt": total_debt,
            "Net Debt": total_debt - cash,
            "Industry": info.get('industry', 'N/A'),
            "Tax Rate": 0.24 
        }
    except: return None

def calculate_unlevered_beta(levered_beta, debt, equity, tax_rate=0.24):
    if equity <= 0: return levered_beta
    return levered_beta / (1 + (1 - tax_rate) * (debt / equity))

def get_historical_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).normalize().tz_localize(None)
        
        # Resampling settimanale con OHLCV completo
        df_res = df.resample("W-FRI").agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
        }).dropna()
        df_res["Var %"] = df_res["Close"].pct_change()
        return df_res
    except: return None

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Configurazione")
search_query = st.sidebar.text_input("Cerca Titolo (es. Enel, Ferrari):", "")

if search_query:
    search_results = search_yahoo_finance(search_query)
    if search_results:
        st.sidebar.selectbox("Risultati:", options=search_results, format_func=lambda x: x[0], key='temp_search_selection')
        st.sidebar.button("➕ Aggiungi al Portafoglio", on_click=add_ticker_to_portfolio, type="primary")

st.sidebar.markdown("---")
bench_dict = {"FTSEMIB.MI": "🇮🇹 Italia (Proxy All-Share)", "^STOXX50E": "🇪🇺 Europa", "^GSPC": "🇺🇸 USA"}
selected_bench = st.sidebar.selectbox("Benchmark di mercato:", list(bench_dict.keys()), format_func=lambda x: bench_dict[x])

rf_input = st.sidebar.number_input("Risk Free (BTP 10Y %)", value=3.8) / 100
mrp_input = st.sidebar.number_input("Market Risk Premium (%)", value=5.5) / 100

# =========================
# LOGICA DI CALCOLO
# =========================
if st.button("🚀 Avvia Analisi Integrata", type="primary"):
    if not st.session_state['selected_tickers_list']:
        st.error("Seleziona almeno un titolo.")
    else:
        full_results = {}
        with st.spinner("Estrazione dati Prezzi (OHLCV) e Bilanci in corso..."):
            df_bench = get_historical_data(selected_bench, "2021-01-01", datetime.date.today().isoformat())
            for t in st.session_state['selected_tickers_list']:
                obj = yf.Ticker(t)
                df_asset = get_historical_data(t, "2021-01-01", datetime.date.today().isoformat())
                fundamentals = get_financial_metrics(obj)
                if df_asset is not None and df_bench is not None:
                    common = df_asset.index.intersection(df_bench.index)
                    y, x = df_asset.loc[common, "Var %"].dropna(), df_bench.loc[common, "Var %"].dropna()
                    common_clean = y.index.intersection(x.index)
                    b_levered = np.cov(y.loc[common_clean], x.loc[common_clean])[0][1] / np.var(x.loc[common_clean])
                    b_unlevered = calculate_unlevered_beta(b_levered, fundamentals['Total Debt'], fundamentals['Market Cap']) if fundamentals else b_levered
                    
                    full_results[t] = {
                        "df": df_asset.loc[common_clean], 
                        "df_bench": df_bench.loc[common_clean],
                        "metrics": {"Beta Levered": b_levered, "Beta Unlevered": b_unlevered, "ATECO": map_industry_to_ateco(fundamentals['Industry']), **fundamentals}, 
                        "financials": obj.balance_sheet
                    }
        st.session_state['analysis_done'] = full_results

# =========================
# GENERAZIONE EXCEL (LAYOUT AVANZATO)
# =========================
def generate_pro_excel(res_dict):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # 1. Foglio Sintesi
        summary_data = []
        for k, v in res_dict.items():
            summary_data.append({
                "Società": st.session_state['ticker_names_map'][k],
                "Ticker": k,
                "Settore ATECO/NACE": v['metrics']['ATECO'],
                "Beta Levered (Mercato)": round(v['metrics']['Beta Levered'], 3),
                "Beta Unlevered (Asset)": round(v['metrics']['Beta Unlevered'], 3),
                "D/E Ratio": round(v['metrics']['Total Debt'] / v['metrics']['Market Cap'], 4) if v['metrics']['Market Cap'] > 0 else 0
            })
        df_sum = pd.DataFrame(summary_data)
        df_sum.to_excel(writer, sheet_name="Sintesi_Rischio", index=False, startrow=1)
        
        ws = writer.sheets["Sintesi_Rischio"]
        ws.cell(row=1, column=1, value="REPORT SINTESI RISCHIO E STRUTTURA CAPITALE").font = Font(bold=True, size=14)
        
        # Note metodologiche in calce
        start_note = len(df_sum) + 5
        ws.cell(row=start_note, column=1, value="METODOLOGIA DI CALCOLO").font = Font(bold=True)
        ws.cell(row=start_note+1, column=1, value=f"Beta Levered (Bl): Cov(Ri, Rm) / Var(Rm) - Regressione lineare su dati settimanali.")
        ws.cell(row=start_note+2, column=1, value=f"Beta Unlevered (Bu): Bl / [1 + (1 - T) * (D/E)] - Formula di Hamada per scorporare il debito.")
        
        # Dettaglio valori grezzi per ogni ticker
        curr_row = start_note + 4
        for k, v in res_dict.items():
            m = v['metrics']
            ws.cell(row=curr_row, column=1, value=f"DETTAGLIO {k}").font = Font(bold=True)
            ws.cell(row=curr_row+1, column=1, value=f"Equity (Market Cap): {m['Market Cap']:,.0f} €")
            ws.cell(row=curr_row+2, column=1, value=f"Debito Totale: {m['Total Debt']:,.0f} €")
            ws.cell(row=curr_row+3, column=1, value=f"Tax Rate (T): {m['Tax Rate']*100}%")
            curr_row += 5

        # 2. Fogli Dati Storici (OHLCV)
        for ticker, data in res_dict.items():
            sheet_name = ticker.replace(".MI", "")[:25] + "_Dati"
            # Preparazione dati speculari (Asset vs Bench)
            df_hist = data['df'].copy().reset_index()
            df_hist.columns = ["Data", "Ap.", "Max", "Min", "Chiusura", "Vol.", "Var %"]
            df_hist.to_excel(writer, sheet_name=sheet_name, index=False)
            
        # 3. Bilancio
        for ticker, data in res_dict.items():
            if not data['financials'].empty:
                data['financials'].to_excel(writer, sheet_name=ticker.replace(".MI", "")[:25] + "_BS")

        # --- FIX DEFINITIVO CANCELLETTI E CENTRATURA ---
        for sheetname in writer.sheets:
            ws_active = writer.sheets[sheetname]
            for col in ws_active.columns:
                max_length = 0
                column = col[0].column_letter
                for cell in col:
                    try:
                        if cell.value:
                            # Se è una data, diamo spazio extra
                            if isinstance(cell.value, (datetime.date, datetime.datetime)):
                                max_length = max(max_length, 20)
                            else:
                                max_length = max(max_length, len(str(cell.value)))
                    except: pass
                # Aumentiamo il fattore di scala e aggiungiamo un margine fisso
                ws_active.column_dimensions[column].width = (max_length * 1.2) + 5
                for cell in col:
                    cell.alignment = Alignment(horizontal='center', vertical='center')

    return output.getvalue()

# =========================
# UI OUTPUT
# =========================
if st.session_state.get('analysis_done'):
    res = st.session_state['analysis_done']
    st.subheader("📋 Sintesi Risultati")
    
    # Visualizzazione Tabella
    summary_list = []
    for k, v in res.items():
        summary_list.append({
            "Società": st.session_state['ticker_names_map'][k],
            "Beta Levered": round(v['metrics']['Beta Levered'], 3),
            "Beta Unlevered": round(v['metrics']['Beta Unlevered'], 3),
            "D/E Ratio": round(v['metrics']['Total Debt'] / v['metrics']['Market Cap'], 2) if v['metrics']['Market Cap'] > 0 else 0
        })
    st.table(pd.DataFrame(summary_list))

    excel_file = generate_pro_excel(res)
    st.download_button("📥 Scarica Analisi Completa (.xlsx)", data=excel_file, file_name="Analisi_Finanziaria_Pro.xlsx", type="primary")

# RELAZIONE METODOLOGICA
st.markdown("---")
st.header("📚 Documentazione Metodologica")
with st.expander("📖 Approfondimento Formule e Processo"):
    st.markdown(r"""
    ### 1. Beta Levered ($\beta_L$)
    Misura il rischio totale dell'azionista (operativo + finanziario). È calcolato come:
    $$ \beta_L = \frac{Cov(R_i, R_m)}{Var(R_m)} $$
    
    ### 2. Beta Unlevered ($\beta_U$)
    Rappresenta il rischio del solo business (Asset Beta), depurato dal debito. Si ottiene tramite la formula di Hamada:
    $$ \beta_U = \frac{\beta_L}{1 + (1 - T) \times \frac{D}{E}} $$
    Dove $D$ è il Debito Totale, $E$ l'Equity (Market Cap) e $T$ l'aliquota fiscale.
    
    ### 3. Interpretazione
    Se $\beta_L$ è molto più alto di $\beta_U$, significa che gran parte della volatilità del titolo è causata dal suo indebitamento piuttosto che dalla natura del suo mercato.
    """)
    