import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import onnxruntime as ort
import json
import os

# --- KONFIGURASI PATH ---
BASE_DIR = "model_outputs"
ONNX_PATH = os.path.join(BASE_DIR, "milk_lstm.onnx")
PARAMS_PATH = os.path.join(BASE_DIR, "scaler_params.json")
DATASET_PATH = "dataset/dataset_sapi_lstm.csv" 

class MilkForecasterONNX:
    def __init__(self, root):
        self.root = root
        self.root.title("🐄 Milk Predictor (ONNX Version)")
        self.root.geometry("500x600")

        try:
            # Load metadata & dataset
            with open(PARAMS_PATH, 'r') as f:
                self.params = json.load(f)
            
            # Load ONNX Session
            self.ort_session = ort.InferenceSession(ONNX_PATH)
            
            self.df = pd.read_csv(DATASET_PATH)
            self.df = self.df[self.df['jumlah_susu'] > 0]
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memuat resource: {e}")
            root.destroy()

        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Prediksi Produksi Susu (ONNX)", font=("Arial", 14, "bold")).pack(pady=10)

        ttk.Label(main_frame, text="Pilih ID Sapi:").pack(anchor="w")
        self.cow_ids = sorted(self.df['sapi_id'].unique().tolist())
        self.cow_combo = ttk.Combobox(main_frame, values=self.cow_ids, state="readonly")
        self.cow_combo.pack(fill=tk.X, pady=5)
        self.cow_combo.set("Pilih Sapi...")

        ttk.Label(main_frame, text="Jumlah Hari Prediksi:").pack(anchor="w", pady=(10, 0))
        self.days_entry = ttk.Entry(main_frame)
        self.days_entry.pack(fill=tk.X, pady=5)
        self.days_entry.insert(0, "7")

        self.predict_btn = ttk.Button(main_frame, text="🚀 Jalankan Prediksi", command=self.run_forecasting)
        self.predict_btn.pack(fill=tk.X, pady=20)

        self.result_text = tk.Text(main_frame, height=15, state="disabled", font=("Courier", 10))
        self.result_text.pack(fill=tk.BOTH, expand=True)

    def run_forecasting(self):
        cow_id = self.cow_combo.get()
        if cow_id == "Pilih Sapi...": return

        # 1. Ambil data spesifik sapi tersebut
        cow_data = self.df[self.df['sapi_id'] == cow_id].copy()
        
        # --- TAMBAHKAN PROSES INI (Feature Engineering Manual) ---
        # Buat kolom waktu yang hilang
        cow_data['tgl_pemerahan'] = pd.to_datetime(cow_data['tgl_pemerahan'])
        cow_data['bulan'] = cow_data['tgl_pemerahan'].dt.month
        cow_data['hari_minggu'] = cow_data['tgl_pemerahan'].dt.dayofweek
        cow_data['jam'] = cow_data['tgl_pemerahan'].dt.hour
        
        # Buat kolom encoding (mapping dari scaler_params.json)
        enc_map = self.params['categorical_encodings']
        for col in ['jenis_sapi', 'jenis_pakan', 'kondisi_sapi', 'status_reproduksi']:
            # Gunakan mapping yang tersimpan untuk mengubah text menjadi angka
            cow_data[f'{col}_enc'] = cow_data[col].astype(str).str.lower().str.strip().map(enc_map[col]).fillna(0)
        
        cow_data = cow_data.sort_values('tgl_pemerahan')
        # ---------------------------------------------------------

        # 2. Ambil 14 data terakhir berdasarkan feature_names yang ada di JSON
        feature_cols = self.params['feature_names']
        try:
            last_features = cow_data[feature_cols].tail(self.params['window']).values.astype(np.float32)
        except KeyError as e:
            messagebox.showerror("Error Kolom", f"Kolom berikut tidak ditemukan di GUI: {e}")
            return

        # 3. Scaling Input
        x_min, x_max = np.array(self.params['X_min']), np.array(self.params['X_max'])
        current_window = (last_features - x_min) / (x_max - x_min + 1e-7)

        predictions = []
        total_milk = 0
        try:
            days = int(self.days_entry.get())
        except:
            days = 7

        # 4. Recursive Loop Prediksi
        for _ in range(days):
            input_name = self.ort_session.get_inputs()[0].name
            input_data = current_window.reshape(1, self.params['window'], self.params['n_features']).astype(np.float32)
            pred_scaled = self.ort_session.run(None, {input_name: input_data})[0][0][0]

            y_min, y_max = self.params['y_min'], self.params['y_max']
            pred_liter = max(0, pred_scaled * (y_max - y_min) + y_min)
            
            predictions.append(pred_liter)
            total_milk += pred_liter

            # Update Window (Rolling)
            new_row = current_window[-1].copy()
            new_row[0] = pred_scaled 
            current_window = np.roll(current_window, -1, axis=0)
            current_window[-1] = new_row

        self.display_results(predictions, total_milk)

    def display_results(self, preds, total):
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"{'Hari':<8} | {'Prediksi (L)':<12}\n" + "-"*25 + "\n")
        for i, p in enumerate(preds, 1):
            self.result_text.insert(tk.END, f"Hari {i:<3} | {p:>10.2f} L\n")
        self.result_text.insert(tk.END, "-"*25 + "\n")
        self.result_text.insert(tk.END, f"TOTAL: {total:.2f} Liter")
        self.result_text.config(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = MilkForecasterONNX(root)
    root.mainloop()