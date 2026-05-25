import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import onnxruntime as ort
import json
import os

# --- KONFIGURASI PATH ---
BASE_DIR = "model_outputs/augmented_2"
ONNX_PATH = os.path.join(BASE_DIR, "milk_lstm.onnx")
PARAMS_PATH = os.path.join(BASE_DIR, "scaler_params.json")
DATASET_PATH = "dataset/dataset_sapi_lstm_1000.csv" 

class MilkForecasterONNX:
    def __init__(self, root):
        self.root = root
        self.root.title("🐄 Milk Predictor (Robust LSTM Version)")
        self.root.geometry("650x700") # Made wider to fit the new column

        try:
            # 1. Load metadata
            with open(PARAMS_PATH, 'r') as f:
                self.params = json.load(f)
            
            # 2. Load ONNX Session
            self.ort_session = ort.InferenceSession(ONNX_PATH)
            
            # 3. Load Dataset safely (Keeping all records including 0 milk for Hamil status)
            self.df = pd.read_csv(DATASET_PATH)
            
            # Safely drop text columns not needed for the math to prevent errors
            drop_cols = ['nama_sapi', 'pemerahan_id', 'tgl_lahir', 'pemerah']
            self.df = self.df.drop(columns=[c for c in drop_cols if c in self.df.columns], errors='ignore')
            
        except Exception as e:
            messagebox.showerror("Error Init", f"Gagal memuat resource: {e}\nPastikan file ONNX, JSON, dan CSV ada.")
            root.destroy()
            return

        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Prediksi Produksi Susu (ONNX)", font=("Arial", 14, "bold")).pack(pady=10)

        ttk.Label(main_frame, text="Pilih ID Sapi:").pack(anchor="w")
        
        # Get unique cows for dropdown
        self.cow_ids = sorted(self.df['sapi_id'].unique().tolist())
        self.cow_combo = ttk.Combobox(main_frame, values=self.cow_ids, state="readonly")
        self.cow_combo.pack(fill=tk.X, pady=5)
        self.cow_combo.set("Pilih Sapi...")

        ttk.Label(main_frame, text="Jumlah Hari Prediksi:").pack(anchor="w", pady=(10, 0))
        self.days_entry = ttk.Entry(main_frame)
        self.days_entry.pack(fill=tk.X, pady=5)
        self.days_entry.insert(0, "7")

        ttk.Label(main_frame, text="Skenario Pembagian Pakan:").pack(anchor="w", pady=(10, 0))
        self.pakan_mode = ttk.Combobox(main_frame, values=["1. Gunakan Rata-rata 30 Hari Terakhir", "2. Bagi Sama Rata (Input Manual)"], state="readonly")
        self.pakan_mode.pack(fill=tk.X, pady=5)
        self.pakan_mode.set("1. Gunakan Rata-rata 30 Hari Terakhir")

        ttk.Label(main_frame, text="Total Pakan (Hanya jika memilih Opsi 2):").pack(anchor="w")
        self.pakan_entry = ttk.Entry(main_frame)
        self.pakan_entry.pack(fill=tk.X, pady=5)
        self.pakan_entry.insert(0, "350")

        self.predict_btn = ttk.Button(main_frame, text="🚀 Jalankan Prediksi", command=self.run_forecasting)
        self.predict_btn.pack(fill=tk.X, pady=20)

        self.result_text = tk.Text(main_frame, height=18, state="disabled", font=("Courier", 10))
        self.result_text.pack(fill=tk.BOTH, expand=True)

    def run_forecasting(self):
        cow_id = self.cow_combo.get()
        if cow_id == "Pilih Sapi...": return

        # 1. Ambil data spesifik sapi tersebut
        cow_data = self.df[self.df['sapi_id'] == cow_id].copy()
        
        if len(cow_data) == 0:
            messagebox.showwarning("Kosong", "Sapi ini belum memiliki data sama sekali. Tidak bisa diprediksi.")
            return

        # 2. Feature Engineering (Waktu)
        cow_data['tgl_pemerahan'] = pd.to_datetime(cow_data['tgl_pemerahan'])
        cow_data['bulan'] = cow_data['tgl_pemerahan'].dt.month
        cow_data['hari_minggu'] = cow_data['tgl_pemerahan'].dt.dayofweek
        cow_data['jam'] = cow_data['tgl_pemerahan'].dt.hour
        
        # 3. Categorical Encoding
        enc_map = self.params['categorical_encodings']
        for col in ['jenis_sapi', 'jenis_pakan', 'kondisi_sapi', 'status_reproduksi']:
            if col in cow_data.columns:
                clean_series = cow_data[col].astype(str).str.title().str.strip()
                cow_data[f'{col}_enc'] = clean_series.map(enc_map.get(col, {})).fillna(0)
        
        cow_data = cow_data.sort_values('tgl_pemerahan')

        # 4. Ambil [Window] data terakhir berdasarkan feature_names dari JSON
        feature_cols = self.params['feature_names']
        try:
            last_features = cow_data[feature_cols].tail(self.params['window']).values.astype(np.float32)
        except KeyError as e:
            messagebox.showerror("Error Kolom", f"Struktur data tidak sesuai dengan model. Kolom hilang: {e}")
            return

        # 5. Scaling Input Window
        x_min, x_max = np.array(self.params['X_min']), np.array(self.params['X_max'])
        current_window = (last_features - x_min) / (x_max - x_min + 1e-7)

        predictions = []
        status_history = []
        pakan_history = [] # NEW TRACKER
        total_milk = 0
        
        try:
            days = int(self.days_entry.get())
        except ValueError:
            days = 7

        # 6. Recursive Forecasting Loop Setup
        input_name = self.ort_session.get_inputs()[0].name
        last_unscaled_row = cow_data[feature_cols].iloc[-1].copy()
        last_date = cow_data['tgl_pemerahan'].max()
        
        # Invert status mapping to display friendly text labels
        inv_status_map = {int(v): k for k, v in enc_map.get('status_reproduksi', {}).items()}
        status_to_enc = enc_map.get('status_reproduksi', {})
        
        # Initialize tracking for current status
        current_status_enc = int(last_unscaled_row['status_reproduksi_enc'])
        status_text = inv_status_map.get(current_status_enc, "Laktasi")
        
        # Count how many days the cow has already been in this status based on recent history
        recent_history = cow_data.tail(self.params['window'])
        days_in_status = len(recent_history[recent_history['status_reproduksi'] == status_text])

        # --- FEED CALCULATION LOGIC ---
        mode = self.pakan_mode.get()
        if "1" in mode:
            daily_pakan = recent_history['volume_pakan'].mean()
        else:
            try:
                total_pakan = float(self.pakan_entry.get())
                daily_pakan = total_pakan / days
                if daily_pakan < 20:
                    messagebox.showwarning("Peringatan ML", f"Pakan harian ({daily_pakan:.1f} kg) di bawah batas training 20 kg. Prediksi LSTM mungkin anjlok!")
            except ValueError:
                messagebox.showerror("Error", "Input total pakan tidak valid! Jatuh kembali ke rata-rata.")
                daily_pakan = recent_history['volume_pakan'].mean()

        for i in range(days):
            input_data = current_window.reshape(1, self.params['window'], self.params['n_features']).astype(np.float32)
            
            # Run Inference via ONNX
            pred_scaled = self.ort_session.run(None, {input_name: input_data})[0][0][0]

            # Inverse Scale Target Prediction
            y_min, y_max = self.params['y_min'], self.params['y_max']
            pred_liter = max(0.0, float(pred_scaled * (y_max - y_min) + y_min))
            
            # Strict Post-processing Rule: If simulated status is Hamil, milk is 0
            if status_text.lower() == 'hamil':
                pred_liter = 0.0
                
            predictions.append(pred_liter)
            total_milk += pred_liter
            status_history.append(status_text)
            pakan_history.append(daily_pakan) # TRACKING THE FEED USED

            # --- BIOLOGICAL CYCLE SIMULATION ---
            days_in_status += 1
            if status_text == 'Laktasi' and days_in_status > 300:
                status_text = 'Hamil'
                days_in_status = 1
            elif status_text == 'Hamil' and days_in_status > 60:
                status_text = 'Laktasi'
                days_in_status = 1
            
            # Get the new encoded value for the next step
            current_status_enc = status_to_enc.get(status_text, current_status_enc)

            # 7. Roll forward and construct next day's features correctly
            next_date = last_date + pd.Timedelta(days=i+1)
            next_unscaled_row = last_unscaled_row.copy()
            next_unscaled_row['bulan'] = next_date.month
            next_unscaled_row['hari_minggu'] = next_date.dayofweek
            next_unscaled_row['status_reproduksi_enc'] = current_status_enc
            
            # Apply the chosen feed
            next_unscaled_row['volume_pakan'] = daily_pakan
            
            # Scale the newly constructed feature row
            next_scaled_row = (next_unscaled_row.values - x_min) / (x_max - x_min + 1e-7)
            
            # Update the lookback sequence window
            current_window = np.roll(current_window, -1, axis=0)
            current_window[-1] = next_scaled_row

        self.display_results(predictions, status_history, pakan_history, total_milk)

    def display_results(self, preds, statuses, pakans, total):
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tk.END)
        # Add Pakan column and adjust widths
        self.result_text.insert(tk.END, f"{'Hari':<8} | {'Status':<12} | {'Pakan (Kg)':<12} | {'Prediksi Susu (L)':<15}\n")
        self.result_text.insert(tk.END, "-"*58 + "\n")
        for i, (p, s, feed) in enumerate(zip(preds, statuses, pakans), 1):
            self.result_text.insert(tk.END, f"Hari {i:<3} | {s:<12} | {feed:>10.2f} | {p:>15.2f} L\n")
        self.result_text.insert(tk.END, "-"*58 + "\n")
        self.result_text.insert(tk.END, f"TOTAL ESTIMASI SUSU: {total:.2f} Liter")
        self.result_text.config(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = MilkForecasterONNX(root)
    root.pack_propagate(False)
    root.mainloop()