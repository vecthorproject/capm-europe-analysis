import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import io 
import requests 
import datetime
from openpyxl.styles import Alignment, Font, PatternFill

# =========================
# CONFIGURAZIONE APP
# =========================
st.set_page_config(page_title="Analisi Beta Pro (5Y)", layout="wide")
st.title("📊 Analisi Finanziaria Integrata: Prezzi & Bilanci (5 Anni)")
st.markdown("""
Strumento professionale per il calcolo del **Beta (5Y Weekly)** e l'analisi della struttura del capitale.
I dati fondamentali sono estratti in tempo reale per calcolare il **Beta Unlevered** (Rischio Asset).
""")

# =========================
# GESTIONE STATO (Session State)
# =========================
if 'selected_tickers_list' not in st.session_state: st.session_state['selected_tickers_list'] = [] 
if 'ticker_names_map' not in st.session_state: st.session_state['ticker_names_map'] = {}

# =========================
# HELPERS & MAPPATURA
# =========================
def map_industry_to_ateco(industry):
    mapping = {
        "Banks—Diversified": "64.19 (Banche)", "Auto Manufacturers": "29.10 (Auto)",
        "Luxury Goods": "47.71 (Lusso)", "Utilities—Renewable": "35.11 (Energia)",
        "Oil & Gas Integrated": "06.10 (Oil&Gas)", "Semiconductors": "26.11 (Chip)",
        "Aerospace & Defense": "30.30 (Aerospaziale)", "Insurance—Diversified": "65.12 (Assicurazioni)"
    }
    return mapping.get(industry, "N.D.")

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
                sym = item.get('symbol')
                name = item.get('shortname') or item.get('longname')
                exch = item.get('exchange')
                if sym and name:
                    results.append((f"{name} ({sym}) - {exch}", sym, name))
        return results
    except: return []

def add_ticker():
    sel = st.session_state.get('temp_search_selection')
    if sel:
        sym, name = sel[1], sel[2]
        if sym not in st.session_state['selected_tickers_list']:
            st.session_state['selected_tickers_list'].append(sym)
            st.session_state['ticker_names_map'][sym] = name

# =========================
# 1. MOTORE DATI (Robustezza Totale)
# =========================
def get_financials_robust(ticker_obj):
    """Estrae dati fondamentali con logica a cascata per evitare zeri"""
    try:
        info = ticker_obj.info
        bs = ticker_obj.balance_sheet
        
        # 1. Equity (Market Cap)
        mkt_cap = info.get('marketCap', 0)
        
        # 2. Debito Totale (Ricerca Aggressiva)
        total_debt = 0.0
        
        # A. Da Info (spesso più affidabile per summary)
        if 'totalDebt' in info and info['totalDebt'] is not None:
             total_debt = float(info['totalDebt'])
        
        # B. Se A fallisce, provo dal Bilancio
        if total_debt == 0 and not bs.empty:
            if 'Total Debt' in bs.index:
                total_debt = float(bs.loc['Total Debt'].iloc[0])
            elif 'Long Term Debt' in bs.index:
                lt = float(bs.loc['Long Term Debt'].iloc[0])
                st_debt = float(bs.loc['Current Debt'].iloc[0]) if 'Current Debt' in bs.index else 0.0
                total_debt = lt + st_debt
        
        # 3. Tax Rate (Standard 24% se non disponibile)
        tax_rate = 0.24

        return {
            "Market Cap": mkt_cap,
            "Total Debt": total_debt,
            "Industry": info.get('industry', 'N.D.'),
            "Tax Rate": tax_rate
        }
    except:
        return None

def get_market_data_5y(ticker):
    """Scarica 5 anni di dati settimanali"""
    try:
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(weeks=260) # 5 Anni esatti
        
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty: return None
        
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # Resampling Settimanale (Venerdì)
        df_wk = df.resample('W-FRI').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
        df_wk['Var %'] = df_wk['Close'].pct_change()
        return df_wk
    except: return None

# =========================
# UI SIDEBAR
# =========================
st.sidebar.header("⚙️ Configurazione")
q = st.sidebar.text_input("Cerca Ticker (es. RACE.MI):", "")
if q:
    res = search_yahoo_finance(q)
    if res:
        st.sidebar.selectbox("Risultati:", res, format_func=lambda x: x[0], key='temp_search_selection')
        st.sidebar.button("➕ Aggiungi", on_click=add_ticker, type="primary")

bench_map = {"FTSEMIB.MI": "Italia (FTSE MIB)", "^STOXX50E": "Europa (Stoxx 50)", "^GSPC": "USA (S&P 500)"}
sel_bench = st.sidebar.selectbox("Benchmark:", list(bench_map.keys()), format_func=lambda x: bench_map[x])

st.sidebar.markdown("---")
st.sidebar.info("📅 Finestra temporale: **5 Anni**\n\nFrequenza: **Settimanale**")

# =========================
# CORE ANALISI
# =========================
if st.button("🚀 Avvia Calcolo (5 Anni)", type="primary"):
    if not st.session_state['selected_tickers_list']:
        st.error("Lista titoli vuota.")
    else:
        results = {}
        with st.spinner("Analisi su 5 anni in corso..."):
            df_bench = get_market_data_5y(sel_bench)
            
            for t in st.session_state['selected_tickers_list']:
                df_ass = get_market_data_5y(t)
                fund = get_financials_robust(yf.Ticker(t))
                
                if df_ass is not None and df_bench is not None:
                    # Allineamento Temporale Rigoroso
                    common = df_ass.index.intersection(df_bench.index)
                    if len(common) > 50: # Minimo statistico per 5 anni
                        y = df_ass.loc[common, 'Var %'].dropna()
                        x = df_bench.loc[common, 'Var %'].dropna()
                        idx_clean = y.index.intersection(x.index)
                        
                        # 1. Beta Levered (Regressione)
                        cov = np.cov(y.loc[idx_clean], x.loc[idx_clean])[0][1]
                        var = np.var(x.loc[idx_clean])
                        beta_lev = cov / var if var != 0 else 0
                        
                        # 2. Beta Unlevered (Hamada)
                        beta_unlev = beta_lev
                        d_e_ratio = 0.0
                        equity = 0
                        debt = 0
                        
                        if fund:
                            equity = fund['Market Cap']
                            debt = fund['Total Debt']
                            if equity > 0:
                                d_e_ratio = debt / equity
                                # Bu = Bl / [1 + (1-T)*D/E]
                                beta_unlev = beta_lev / (1 + (1 - fund['Tax Rate']) * d_e_ratio)
                        
                        metrics = {
                            "Beta Lev": beta_lev, "Beta Unlev": beta_unlev,
                            "D/E Ratio": d_e_ratio, "Equity": equity, "Debt": debt,
                            "ATECO": map_industry_to_ateco(fund.get('Industry') if fund else 'N.D.')
                        }
                        
                        results[t] = {
                            "df": df_ass.loc[idx_clean],
                            "metrics": metrics,
                            "bs": yf.Ticker(t).balance_sheet
                        }
        
        st.session_state['res'] = results

# =========================
# OUTPUT & EXCEL (Fix Formattazione Totale)
# =========================
if 'res' in st.session_state:
    data = st.session_state['res']
    
    # Sintesi a video
    rows = []
    for k, v in data.items():
        m = v['metrics']
        rows.append({
            "Società": st.session_state['ticker_names_map'][k],
            "Beta Lev (5Y)": f"{m['Beta Lev']:.3f}",
            "Beta Unlev": f"{m['Beta Unlev']:.3f}",
            "D/E (Mkt)": f"{m['D/E Ratio']:.4f}",
            "Settore": m['ATECO']
        })
    
    st.subheader("📋 Sintesi Analisi (5 Anni)")
    st.table(pd.DataFrame(rows))

    # Generazione Excel
    def generate_perfect_excel(res_data):
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine='openpyxl') as writer:
            
            # --- FOGLIO 1: SINTESI ---
            df_s = pd.DataFrame(rows)
            df_s.to_excel(writer, sheet_name="Sintesi", index=False, startrow=1)
            ws = writer.sheets["Sintesi"]
            
            # Intestazione
            ws.cell(1, 1, "REPORT ANALISI RISCHIO (5 ANNI WEEKLY)").font = Font(bold=True, size=14, color="000080")
            
            # Note Metodologiche
            r_start = len(df_s) + 5
            ws.cell(r_start, 1, "METODOLOGIA E DETTAGLIO DATI").font = Font(bold=True)
            ws.cell(r_start+1, 1, "1. Finestra Temporale: 5 Anni (260 Settimane).")
            ws.cell(r_start+2, 1, "2. D/E Ratio = Total Debt / Market Cap (Valori di Mercato).")
            ws.cell(r_start+3, 1, "3. Formula Hamada: Bu = Bl / [1 + (0.76 * D/E)].")
            
            # Dettaglio Valori Grezzi
            curr = r_start + 5
            for k, v in res_data.items():
                m = v['metrics']
                ws.cell(curr, 1, f"Dati {k}:").font = Font(bold=True)
                ws.cell(curr+1, 1, f"- Equity: {m['Equity']:,.0f} €")
                ws.cell(curr+2, 1, f"- Debito Totale: {m['Debt']:,.0f} €")
                curr += 4

            # --- FOGLI 2+: DATI STORICI ---
            for t, d in res_data.items():
                # Dati OHLCV
                df_exp = d['df'].copy().reset_index()
                # Rinomina colonne per pulizia
                df_exp.columns = ["Data Rilevazione", "Apertura", "Massimo", "Minimo", "Chiusura", "Volume", "Var %"]
                df_exp['Data Rilevazione'] = df_exp['Data Rilevazione'].dt.date # Via l'orario
                
                sheet_name = f"{t.replace('.MI','')[:20]}_Dati"
                df_exp.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Bilancio (se c'è)
                if not d['bs'].empty:
                    d['bs'].to_excel(writer, sheet_name=f"{t.replace('.MI','')[:20]}_BS")

            # --- FORMATTAZIONE "NO ###" (BODYGUARD) ---
            for sheet in writer.sheets:
                ws_cur = writer.sheets[sheet]
                for col in ws_cur.columns:
                    col_letter = col[0].column_letter
                    header_txt = str(col[0].value).lower()
                    
                    # Logica larghezza
                    if "data" in header_txt or "date" in header_txt:
                        final_width = 22 # Larghezza fissa sicura per Date
                    else:
                        max_len = 0
                        for cell in col:
                            try:
                                if cell.value: max_len = max(max_len, len(str(cell.value)))
                            except: pass
                        final_width = min(max_len * 1.2 + 2, 60) # Moltiplicatore + Tetto max
                    
                    ws_cur.column_dimensions[col_letter].width = final_width
                    
                    # Allineamento Centrato
                    for cell in col:
                        cell.alignment = Alignment(horizontal='center', vertical='center')

        return bio.getvalue()

    st.download_button("📥 Scarica Analisi Completa (Excel)", data=generate_perfect_excel(data), file_name="Analisi_Beta_5Y_Pro.xlsx", type="primary")
    