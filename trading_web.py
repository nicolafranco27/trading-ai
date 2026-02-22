import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# Configurazione base della pagina (adatta anche per mobile)
st.set_page_config(page_title="AI Quant Trader PRO", page_icon="📈", layout="wide")

st.title("📈 AI Quant Trader PRO")
st.markdown("### Trend Intelligence - Versione Mobile / Web")
st.write("Questa versione ti permette di accedere alle analisi dal tuo smartphone sul browser.")

# --- SETTINGS ---
with st.expander("⚙️ IMPOSTAZIONI ANALISI", expanded=True):
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        budget = st.number_input("Budget (€)", min_value=10.0, value=1000.0, step=100.0)
    with col_s2:
        option = st.selectbox("Timeframe", options=[("Minuti", "m"), ("Ore", "h"), ("Giorni", "d")], index=2, format_func=lambda x: x[0])
        timeframe = option[1]
    with col_s3:
        horizon = st.number_input("Orizzonte (Periodi futuri)", min_value=1, value=7, step=1)

    run_button = st.button("🚀 ANALIZZA TREND E PROBABILITÀ", use_container_width=True, type="primary")

def get_ai_data(ticker, q, unita, budget):
    try:
        p = "1d" if unita=="m" else "7d" if unita=="h" else "2y"
        i = "1m" if unita=="m" else "60m" if unita=="h" else "1d"
        df = yf.download(ticker, period=p, interval=i, progress=False, auto_adjust=True)
        
        if df.empty:
            return f"Nessun dato per {ticker}", "white"
            
        y = df['Close'].values.flatten()
        y = y[~np.isnan(y)]
        
        if len(y) < 50:
            return f"Dati insufficienti per {ticker} (min 50 periodi)", "white"
            
        X = np.arange(len(y)).reshape(-1, 1)
        
        # Addestramento con Foresta Casuale (150 alberi per stabilità)
        model = RandomForestRegressor(n_estimators=150, random_state=42).fit(X, y)
        
        p_attuale = float(y[-1])
        pred = float(model.predict(np.array([[len(y) + q]]))[0])
        
        # --- NUOVA LOGICA PROBABILITÀ REALE ---
        # 1. Calcolo del Rumore (Efficiency Ratio)
        cambio_netto = abs(y[-1] - y[-20])
        somma_cambi = np.sum(np.abs(np.diff(y[-20:])))
        efficiency_ratio = cambio_netto / somma_cambi if somma_cambi != 0 else 0
        
        # 2. Qualità del Trend (R-squared degli ultimi 50 periodi)
        r2_score = model.score(X[-50:], y[-50:])
        
        # 3. Probabilità Finale: combina stabilità storica e pulizia del trend attuale
        prob_reale = (r2_score * 0.7) + (efficiency_ratio * 0.3)
        prob_reale = np.clip(prob_reale * 0.95, 0.1, 0.98) # Margine prudenziale
        
        # Livelli Operativi
        vol = np.std(y[-20:])
        is_buy = pred > p_attuale
        color = "green" if is_buy else "red"
        dir = 1 if is_buy else -1
        investimento = budget * (prob_reale * 0.1)

        def calc_profit(target): return investimento * (((target - p_attuale) / p_attuale) * dir)
        
        tp1, tp2, tp3 = p_attuale+(vol*0.8*dir), p_attuale+(vol*1.5*dir), p_attuale+(vol*2.5*dir)
        sl = p_attuale-(vol*1.2*dir)

        info = (f"LIVE: {datetime.now().strftime('%H:%M:%S')}\n"
                f"==============================\n"
                f"AZIONE: {'BUY 🟢' if is_buy else 'SELL 🔴'}\n"
                f"PROBABILITÀ REALE: {prob_reale*100:.1f}%\n"
                f"INVESTIMENTO:      {investimento:.2f} €\n"
                f"------------------------------\n"
                f"ANALISI DEL TREND:\n"
                f"Qualità (R2): {r2_score:.2f}\n"
                f"Pulizia Trend: {efficiency_ratio:.2f}\n"
                f"------------------------------\n"
                f"ENTRY: {p_attuale:.2f} -> TARGET: {pred:.2f}\n"
                f"------------------------------\n"
                f"TP 1: {tp1:.2f} (+{calc_profit(tp1):.2f}€)\n"
                f"TP 2: {tp2:.2f} (+{calc_profit(tp2):.2f}€)\n"
                f"TP 3: {tp3:.2f} (+{calc_profit(tp3):.2f}€)\n\n"
                f"SL:   {sl:.2f} ({calc_profit(sl):.2f}€)\n"
                f"------------------------------\n"
                f"Nota: Probabilità basata sulla\n"
                f"linearità del movimento attuale.")
        return info, color
    except Exception as e:
        return f"Errore: {str(e)}", "red"

if run_button:
    st.markdown("---")
    
    with st.spinner("Analisi in corso sui mercati... attendi qualche secondo."):
        info_oro, color_oro = get_ai_data("GC=F", horizon, timeframe, budget)
        info_tech, color_tech = get_ai_data("NQ=F", horizon, timeframe, budget)
        info_btc, color_btc = get_ai_data("BTC-USD", horizon, timeframe, budget)
        
    # Da mobile, Streamlit visualizzerà automaticamente le colonne una sotto l'altra.
    col1, col2, col3 = st.columns(3)
    
    def display_asset(col, title, info, color):
        with col:
            st.markdown(f"**{title}**")
            # Mostra la box verde o rossa in base al segnale
            if color == "green":
                st.success("Segnale Rialzista")
            else:
                st.error("Segnale Ribassista")
            # Usiamo st.code per simulare l'effetto monospazio del terminale / interfaccia originale
            st.code(info, language="text")

    display_asset(col1, "ORO 🟡", info_oro, color_oro)
    display_asset(col2, "NASDAQ 🔵", info_tech, color_tech)
    display_asset(col3, "BITCOIN 🟠", info_btc, color_btc)
