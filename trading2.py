import customtkinter as ctk
import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class TradingAIApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Quant Trader PRO - Trend Intelligence")
        self.geometry("1200x820")

        # --- SETTINGS ---
        self.frame_settings = ctk.CTkFrame(self)
        self.frame_settings.pack(pady=20, padx=20, fill="x")
        self.frame_settings.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

        self.label_title = ctk.CTkLabel(self.frame_settings, text="IMPOSTAZIONI ANALISI", font=("Segoe UI", 16, "bold"))
        self.label_title.grid(row=0, column=0, columnspan=5, pady=(15, 10))

        self.entry_budget = ctk.CTkEntry(self.frame_settings, placeholder_text="Budget (€)", width=140)
        self.entry_budget.grid(row=1, column=1, padx=10, pady=15, sticky="ew")

        self.seg_timeframe = ctk.CTkSegmentedButton(self.frame_settings, values=["m", "h", "d"])
        self.seg_timeframe.set("d")
        self.seg_timeframe.grid(row=1, column=2, padx=10, pady=15)

        self.entry_horizon = ctk.CTkEntry(self.frame_settings, placeholder_text="Orizzonte", width=100)
        self.entry_horizon.grid(row=1, column=3, padx=10, pady=15, sticky="ew")

        self.btn_run = ctk.CTkButton(self, text="ANALIZZA TREND E PROBABILITÀ", command=self.run_analysis, font=("Segoe UI", 14, "bold"), height=40, fg_color="#27ae60")
        self.btn_run.pack(pady=10)

        # --- RISULTATI ---
        self.frame_results = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_results.pack(expand=True, fill="both", padx=20, pady=10)
        self.txt_oro = self.create_asset_box(self.frame_results, "ORO 🟡", "left")
        self.txt_tech = self.create_asset_box(self.frame_results, "TECH500 🔵", "left")
        self.txt_btc = self.create_asset_box(self.frame_results, "BITCOIN 🟠", "left")

    def create_asset_box(self, master, title, side):
        box = ctk.CTkFrame(master, corner_radius=15)
        box.pack(side=side, expand=True, fill="both", padx=10, pady=10)
        ctk.CTkLabel(box, text=title, font=("Segoe UI", 18, "bold")).pack(pady=10)
        txt = ctk.CTkTextbox(box, width=320, height=420, font=("Consolas", 12), border_width=2)
        txt.pack(pady=10, padx=15)
        return txt

    def get_ai_data(self, ticker, q, unita, budget):
        try:
            p = "1d" if unita=="m" else "7d" if unita=="h" else "2y"
            i = "1m" if unita=="m" else "60m" if unita=="h" else "1d"
            df = yf.download(ticker, period=p, interval=i, progress=False, auto_adjust=True)
            
            y = df['Close'].values.flatten()
            y = y[~np.isnan(y)]
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
            # Non penalizziamo più il tempo lungo, ma l'inefficienza del movimento
            prob_reale = (r2_score * 0.7) + (efficiency_ratio * 0.3)
            prob_reale = np.clip(prob_reale * 0.95, 0.1, 0.98) # Margine prudenziale
            
            # Livelli Operativi
            vol = np.std(y[-20:])
            is_buy = pred > p_attuale
            color = "#2ecc71" if is_buy else "#e74c3c"
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
            return f"Errore: {str(e)}", "#ffffff"

    def run_analysis(self):
        try:
            b = float(self.entry_budget.get() or 1000)
            u = self.seg_timeframe.get()
            q = int(self.entry_horizon.get() or 7)

            for txt, ticker in [(self.txt_oro, "GC=F"), (self.txt_tech, "NQ=F"), (self.txt_btc, "BTC-USD")]:
                txt.delete("0.0", "end")
                txt.insert("end", "🔄 Analizzando Trend...")
                self.update_idletasks()
                info, col = self.get_ai_data(ticker, q, u, b)
                txt.delete("0.0", "end")
                txt.insert("end", info)
                txt.configure(text_color=col, border_color=col)
        except ValueError: pass

if __name__ == "__main__":
    app = TradingAIApp()
    app.mainloop()