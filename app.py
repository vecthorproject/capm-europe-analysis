import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import io 
import requests 
import datetime
from openpyxl.styles import Alignment, Font

# =========================
# CONFIGURAZIONE APP
# =========================
st.set_page_config(page_title="Analisi Beta Ultimate", layout="wide", page_icon="💼")

# =========================
# GESTIONE STATO (Session State)
# =========================
# Usiamo un dizionario per mantenere ticker e nome pulito
if 'portfolio' not in st.session_state: 
    st.session_state['portfolio'] = {} # Format: {'RACE.MI': 'Ferrari N.V.'}

# =========================
# FUNZIONI UTILI (Helpers)
# =========================
def map_industry_to_ateco(industry):
    mapping = {
        "Banks—Diversified": "64.19 (Banche)", "Auto Manufacturers": "29.10 (Auto)",
        "Luxury Goods": "47.71 (Lusso)", "Utilities—Renewable": "35.11 (Energia)",
        "Oil & Gas Integrated": "06.10 (Oil&Gas)", "Semiconductors": "26.11 (Chip)",
        "Aerospace & Defense": "30.30 (Aerospaziale)", "Insurance—Diversified": "65.12 (Assicurazioni)",
        "Beverages—Wineries & Distilleries": "11.02 (Vini)", "Apparel Manufacturing": "14.13 (Abbigliamento)"
    }
    return mapping.get(industry, "N.D.")

def search_yahoo_finance(query):
    if not query or len(query) < 2: return []
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=3)
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

# =========================
# 1. SIDEBAR: PANNELLO DI CONTROLLO TOTALE
# =========================
st.sidebar.title("⚙️ Impostazioni Analisi")

st.sidebar.subheader("1. Orizzonte Temporale")
time_period = st.sidebar.selectbox(
    "Durata Analisi:", 
    ["3 Anni (Standard)", "5 Anni (Lungo Periodo)", "1 Anno (Breve Periodo)"],
    index=0
)
frequency = st.sidebar.selectbox("Frequenza Dati:", ["Settimanale (Consigliato)", "Mensile"])

# Traduzione input in date e intervalli
end_date = datetime.date.today()
if "5 Anni" in time_period: delta_weeks = 260
elif "3 Anni" in time_period: delta_weeks = 156
else: delta_weeks = 52
start_date = end_date - datetime.timedelta(weeks=delta_weeks)

yf_interval = "1wk"
resample_rule = "W-FRI"
if "Mensile" in frequency:
    yf_interval = "1mo"
    resample_rule = "ME"

st.sidebar.divider()

st.sidebar.subheader("2. Parametri CAPM (Ke)")
rf_input = st.sidebar.number_input("Risk Free Rate (BTP 10Y) %:", value=3.80, step=0.10) / 100
mrp_input = st.sidebar.number_input("Market Risk Premium (MRP) %:", value=5.50, step=0.10, help="Fonte suggerita: Survey Pablo Fernandez (IESE)") / 100

st.sidebar.divider()

st.sidebar.subheader("3. Benchmark")
bench_map = {
    "FTSEMIB.MI": "🇮🇹 Italia (FTSE MIB)", 
    "^STOXX50E": "🇪🇺 Europa (Euro Stoxx 50)", 
    "^GSPC": "🇺🇸 USA (S&P 500)",
    "^GDAXI": "🇩🇪 Germania (DAX)",
    "^FCHI": "🇫🇷 Francia (CAC 40)"
}
sel_bench_code = st.sidebar.selectbox("Indice di Riferimento:", list(bench_map.keys()), format_func=lambda x: bench_map[x])

# =========================
# 2. GESTIONE PORTAFOGLIO (Main Page)
# =========================
st.title("💼 Gestione Portafoglio & Analisi Beta")

col_search, col_add = st.columns([3, 1])

with col_search:
    search_q = st.text_input("🔍 Cerca Titolo (Nome o Ticker):", placeholder="Es. Enel, Intesa, Ferrari...")
    search_res = search_yahoo_finance(search_q) if search_q else []
    
    selected_asset = None
    if search_res:
        selected_asset = st.selectbox("Seleziona dai risultati:", search_res, format_func=lambda x: x[0], label_visibility="collapsed")

with col_add:
    st.write("") # Spacer
    st.write("") # Spacer
    if st.button("➕ Aggiungi", type="primary", use_container_width=True):
        if selected_asset:
            sym, name = selected_asset[1], selected_asset[2]
            if sym not in st.session_state['portfolio']:
                st.session_state['portfolio'][sym] = name
                st.toast(f"✅ {name} aggiunto al portafoglio!", icon="🚀")
            else:
                st.toast(f"⚠️ {name} è già presente!", icon="Duplicato")
        else:
            st.warning("Cerca e seleziona un titolo prima.")

# --- LISTA PORTAFOGLIO CON TASTO RIMUOVI ---
if st.session_state['portfolio']:
    st.markdown("### 📋 Titoli Selezionati")
    
    # Creiamo una griglia per mostrare i titoli e il tasto elimina
    for ticker, name in list(st.session_state['portfolio'].items()):
        c1, c2, c3 = st.columns([0.1, 0.7, 0.2])
        with c1:
            st.write("📌")
        with c2:
            st.write(f"**{name}** ({ticker})")
        with c3:
            if st.button("🗑️ Elimina", key=f"del_{ticker}"):
                del st.session_state['portfolio'][ticker]
                st.rerun() # Ricarica la pagina per aggiornare la lista
    
    st.markdown("---")
else:
    st.info("Il portafoglio è vuoto. Cerca un titolo sopra per iniziare.")

# =========================
# 3. MOTORE DI CALCOLO
# =========================

def get_financials_robust(ticker_obj):
    """Estrae Debito ed Equity in modo robusto"""
    try:
        info = ticker_obj.info
        bs = ticker_obj.balance_sheet
        
        mkt_cap = info.get('marketCap', 0)
        total_debt = 0.0
        
        # 1. Info (Priorità)
        if 'totalDebt' in info and info['totalDebt'] is not None:
             total_debt = float(info['totalDebt'])
        
        # 2. Bilancio (Fallback)
        if total_debt == 0 and not bs.empty:
            if 'Total Debt' in bs.index:
                total_debt = float(bs.loc['Total Debt'].iloc[0])
            elif 'Long Term Debt' in bs.index:
                lt = float(bs.loc['Long Term Debt'].iloc[0])
                st_debt = float(bs.loc['Current Debt'].iloc[0]) if 'Current Debt' in bs.index else 0.0
                total_debt = lt + st_debt
        
        return {
            "Market Cap": mkt_cap,
            "Total Debt": total_debt,
            "Industry": info.get('industry', 'N.D.'),
            "Tax Rate": 0.24 # Standard IT
        }
    except: return None

def get_market_data(ticker, start, end, interval):
    """Scarica dati OHLCV"""
    try:
        df = yf.download(ticker, start=start, end=end, interval="1d", progress=False) # Scarica daily per precisione
        if df.empty: return None
        
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # Resampling personalizzato
        df_res = df.resample(resample_rule).agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
        # Rendimenti Semplici (Standard CAPM)
        df_res['Var %'] = df_res['Close'].pct_change()
        return df_res
    except: return None

# =========================
# 4. ESECUZIONE ANALISI
# =========================

if st.session_state['portfolio']:
    if st.button("🚀 Avvia Analisi Completa", type="primary", use_container_width=True):
        
        results = {}
        with st.spinner(f"Analisi in corso ({start_date} -> {end_date})..."):
            
            # Scarico Benchmark
            df_bench = get_market_data(sel_bench_code, start_date, end_date, yf_interval)
            
            for t, t_name in st.session_state['portfolio'].items():
                df_asset = get_market_data(t, start_date, end_date, yf_interval)
                fund = get_financials_robust(yf.Ticker(t))
                
                if df_asset is not None and df_bench is not None:
                    # Allineamento Date
                    common = df_asset.index.intersection(df_bench.index)
                    if len(common) > 10: # Controllo minimo dati
                        y = df_asset.loc[common, 'Var %'].dropna()
                        x = df_bench.loc[common, 'Var %'].dropna()
                        idx_clean = y.index.intersection(x.index)
                        
                        # 1. Beta Levered (Statistico)
                        cov = np.cov(y.loc[idx_clean], x.loc[idx_clean], ddof=1)[0][1]
                        var = np.var(x.loc[idx_clean], ddof=1)
                        beta_lev = cov / var if var != 0 else 0
                        
                        # 2. Beta Unlevered (Fondamentale - Hamada)
                        beta_unlev = beta_lev
                        d_e_ratio = 0.0
                        
                        if fund:
                            equity = fund['Market Cap']
                            debt = fund['Total Debt']
                            if equity > 0:
                                d_e_ratio = debt / equity
                                beta_unlev = beta_lev / (1 + (1 - fund['Tax Rate']) * d_e_ratio)
                        
                        # 3. Costo Equity (CAPM)
                        # Ke = Rf + Beta * MRP
                        ke = rf_input + (beta_lev * mrp_input)

                        metrics = {
                            "Beta Lev": beta_lev,
                            "Beta Unlev": beta_unlev,
                            "Ke (CAPM)": ke,
                            "D/E Ratio": d_e_ratio,
                            "Equity": fund['Market Cap'] if fund else 0,
                            "Debt": fund['Total Debt'] if fund else 0,
                            "ATECO": map_industry_to_ateco(fund.get('Industry') if fund else 'N.D.')
                        }
                        
                        results[t] = {
                            "df": df_asset.loc[idx_clean],
                            "metrics": metrics,
                            "bs": yf.Ticker(t).balance_sheet,
                            "name": t_name
                        }
        
        # =========================
        # 5. VISUALIZZAZIONE & EXPORT
        # =========================
        if results:
            # Tabella Sintetica
            st.success("Analisi completata con successo!")
            st.subheader("📊 Sintesi Valutazione e Rischio")
            
            summary_rows = []
            for k, v in results.items():
                m = v['metrics']
                summary_rows.append({
                    "Società": v['name'],
                    "Beta Levered": f"{m['Beta Lev']:.2f}",
                    "Ke (CAPM)": f"{m['Ke (CAPM)']*100:.2f}%",
                    "Beta Asset (Unlev)": f"{m['Beta Unlev']:.2f}",
                    "D/E Ratio": f"{m['D/E Ratio']:.2f}",
                    "Settore": m['ATECO']
                })
            
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
            
            # Parametri usati per il calcolo (Trasparenza)
            st.caption(f"ℹ️ Parametri calcolo: Rf={rf_input*100:.2f}% | MRP={mrp_input*100:.2f}% | Periodo={time_period}")

            # Generazione Excel
            def make_excel(res_data):
                bio = io.BytesIO()
                with pd.ExcelWriter(bio, engine='openpyxl') as writer:
                    # FOGLIO SINTESI
                    df_s = pd.DataFrame(summary_rows)
                    df_s.to_excel(writer, sheet_name="Sintesi", index=False, startrow=1)
                    ws = writer.sheets["Sintesi"]
                    
                    # Header Principale
                    title = f"REPORT ANALISI ({time_period} - {frequency})"
                    ws.cell(1, 1, title).font = Font(bold=True, size=14, color="000080")
                    
                    # Parametri
                    r_p = len(df_s) + 5
                    ws.cell(r_p, 1, "PARAMETRI INPUT").font = Font(bold=True)
                    ws.cell(r_p+1, 1, f"Risk Free (Rf): {rf_input*100:.2f}%")
                    ws.cell(r_p+2, 1, f"Market Risk Premium (MRP - Fernandez): {mrp_input*100:.2f}%")
                    ws.cell(r_p+3, 1, f"Benchmark: {bench_map[sel_bench_code]}")
                    
                    # Formule
                    r_f = r_p + 5
                    ws.cell(r_f, 1, "FORMULE").font = Font(bold=True)
                    ws.cell(r_f+1, 1, "Ke (CAPM) = Rf + Beta * MRP")
                    ws.cell(r_f+2, 1, "Bu (Hamada) = Bl / [1 + (1-T)*D/E]")

                    # Dati Grezzi per ticker
                    curr = r_f + 4
                    for k, v in res_data.items():
                        m = v['metrics']
                        ws.cell(curr, 1, f"Dati {v['name']} ({k}):").font = Font(bold=True)
                        ws.cell(curr+1, 1, f"- Equity: {m['Equity']:,.0f} €")
                        ws.cell(curr+2, 1, f"- Debito Totale: {m['Debt']:,.0f} €")
                        curr += 4

                    # FOGLI DETTAGLIO
                    for t, d in res_data.items():
                        # Prezzi
                        df_exp = d['df'].copy().reset_index()
                        df_exp.columns = ["Data", "Apertura", "Massimo", "Minimo", "Chiusura", "Volume", "Var %"]
                        df_exp['Data'] = df_exp['Data'].dt.date
                        safe_name = t.replace(".MI", "").replace(".","")[:20]
                        df_exp.to_excel(writer, sheet_name=f"{safe_name}_Dati", index=False)
                        
                        # Bilancio
                        if not d['bs'].empty:
                            d['bs'].to_excel(writer, sheet_name=f"{safe_name}_BS")
                    
                    # FORMATTAZIONE BODYGUARD
                    for sheet in writer.sheets:
                        ws_cur = writer.sheets[sheet]
                        for col in ws_cur.columns:
                            col_letter = col[0].column_letter
                            head = str(col[0].value).lower()
                            
                            if "data" in head or "date" in head:
                                width = 22
                            else:
                                max_l = 0
                                for cell in col:
                                    try:
                                        if cell.value: max_l = max(max_l, len(str(cell.value)))
                                    except: pass
                                width = min(max_l * 1.2 + 2, 60)
                            
                            ws_cur.column_dimensions[col_letter].width = width
                            for cell in col:
                                cell.alignment = Alignment(horizontal='center', vertical='center')

                return bio.getvalue()

            st.download_button(
                "📥 Scarica Report Excel (Ultimate)", 
                data=make_excel(results), 
                file_name=f"Analisi_Beta_{datetime.date.today()}.xlsx", 
                type="primary",
                use_container_width=True
            )
