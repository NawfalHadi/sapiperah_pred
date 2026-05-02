import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
import onnxruntime as ort
import json
import os
from datetime import timedelta

# --- KONFIGURASI PATH ---
BASE_DIR = "model_outputs"
ONNX_PATH = os.path.join(BASE_DIR, "milk_lstm.onnx")
PARAMS_PATH = os.path.join(BASE_DIR, "scaler_params.json")
FIREBASE_CRED_PATH = "cred/sapiperahapp26-firebase-adminsdk-fbsvc-6c1087eda3.json" # <-- Sesuai file kredensial Anda
LOCAL_CSV_PATH = "local_dataset.csv" # <-- Menggunakan CSV sebagai Lokal DB

class CSVForecaster:
    def __init__(self, root):
        self.root = root
        self.root.title("🐄 Milk Predictor (CSV Sync Mode)")
        self.root.geometry("600x750")

        self.init_firebase()
        
        # Load Model & Params
        try:
            with open(PARAMS_PATH, 'r') as f:
                self.params = json.load(f)
            self.ort_session = ort.InferenceSession(ONNX_PATH)
        except Exception as e:
            messagebox.showwarning("Warning", f"ONNX/JSON model belum dimuat sempurna: {e}")

        self.setup_ui()
        self.refresh_cow_dropdown()

    # ==========================================
    # 1. SETUP FIREBASE
    # ==========================================
    def init_firebase(self):
        try:
            cred = credentials.Certificate(FIREBASE_CRED_PATH)
            firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            self.firebase_ready = True
        except Exception as e:
            self.firebase_ready = False
            print("Firebase belum siap (Cek path JSON kredensial):", e)

    # ==========================================
    # 2. SYNC FIREBASE -> CSV (TanPA Last Time)
    # ==========================================
    def sync_from_firestore(self):
        if not self.firebase_ready:
            messagebox.showerror("Error", "Firebase tidak terhubung.")
            return

        self.sync_btn.config(text="⏳ Mengunduh Semua Data...", state="disabled")
        self.root.update()

        try:
            # Mengambil SEMUA dokumen dari koleksi 'pemerahan' (Tanpa filter waktu)
            docs = self.db.collection('pemerahan').stream()
            
            data_list = []
            for doc in docs:
                data = doc.to_dict()
                data['pemerahan_id'] = doc.id # Menyimpan ID dokumen
                
                # Pastikan jumlah_susu dan volume_pakan dalam bentuk numerik
                data['jumlah_susu'] = float(data.get('jumlah_susu', 0))
                data['volume_pakan'] = float(data.get('volume_pakan', 0))
                
                data_list.append(data)

            if data_list:
                # Buat Dataframe dan timpa/buat file CSV
                df = pd.DataFrame(data_list)
                
                # Mengurutkan berdasarkan tanggal agar rapi
                if 'tgl_pemerahan' in df.columns:
                    df = df.sort_values('tgl_pemerahan')
                    
                df.to_csv(LOCAL_CSV_PATH, index=False)
                
                self.refresh_cow_dropdown()
                messagebox.showinfo("Sync Berhasil", f"{len(df)} baris data berhasil diunduh dan disimpan ke {LOCAL_CSV_PATH}!")
            else:
                messagebox.showinfo("Sync Selesai", "Koleksi di Firestore kosong.")

        except Exception as e:
            messagebox.showerror("Sync Gagal", str(e))
        finally:
            self.sync_btn.config(text="🔄 Sync Semua Data dari Firestore", state="normal")

    # ==========================================
    # 3. UI & PREDIKSI (Membaca dari CSV)
    # ==========================================
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Prediksi Susu (Data CSV Lokal)", font=("Arial", 14, "bold")).pack(pady=5)

        self.sync_btn = ttk.Button(main_frame, text="🔄 Sync Semua Data dari Firestore", command=self.sync_from_firestore)
        self.sync_btn.pack(fill=tk.X, pady=(0, 15))

        ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=10)

        ttk.Label(main_frame, text="Pilih Sapi (Dari File CSV):").pack(anchor="w")
        self.cow_combo = ttk.Combobox(main_frame, state="readonly")
        self.cow_combo.pack(fill=tk.X, pady=5)
        self.cow_combo.set("Pilih Sapi...")

        ttk.Label(main_frame, text="Jumlah Hari Prediksi:").pack(anchor="w", pady=(10, 0))
        self.days_entry = ttk.Entry(main_frame)
        self.days_entry.pack(fill=tk.X, pady=5)
        self.days_entry.insert(0, "7")

        self.predict_btn = ttk.Button(main_frame, text="🚀 Prediksi (Berdasarkan CSV)", command=self.run_forecasting)
        self.predict_btn.pack(fill=tk.X, pady=20)

        self.result_text = tk.Text(main_frame, height=18, state="disabled", font=("Courier", 10))
        self.result_text.pack(fill=tk.BOTH, expand=True)

    def refresh_cow_dropdown(self):
        # Cek apakah CSV sudah ada
        if os.path.exists(LOCAL_CSV_PATH):
            try:
                df = pd.read_csv(LOCAL_CSV_PATH)
                if 'sapi_id' in df.columns:
                    cow_ids = sorted(df[df['jumlah_susu'] > 0]['sapi_id'].dropna().unique().tolist())
                    self.cow_combo['values'] = cow_ids
            except Exception as e:
                print("Gagal membaca CSV untuk dropdown:", e)

    def run_forecasting(self):
        cow_id = self.cow_combo.get()
        if cow_id == "Pilih Sapi..." or not cow_id: return

        if not os.path.exists(LOCAL_CSV_PATH):
            messagebox.showerror("File Tidak Ditemukan", "File CSV belum ada. Lakukan Sync terlebih dahulu.")
            return

        window_size = self.params.get('window', 14)
        
        # 1. BACA DARI FILE CSV
        df = pd.read_csv(LOCAL_CSV_PATH)
        cow_data = df[(df['sapi_id'] == cow_id) & (df['jumlah_susu'] > 0)].copy()
        
        if len(cow_data) < window_size:
            messagebox.showwarning("Data Kurang", f"Data riwayat CSV untuk sapi ini baru {len(cow_data)}. Model butuh minimal {window_size}.")
            return

        # 2. Feature Engineering
        cow_data['tgl_pemerahan'] = pd.to_datetime(cow_data['tgl_pemerahan'])
        cow_data = cow_data.sort_values('tgl_pemerahan')
        
        cow_data['bulan'] = cow_data['tgl_pemerahan'].dt.month
        cow_data['hari_minggu'] = cow_data['tgl_pemerahan'].dt.dayofweek
        cow_data['jam'] = cow_data['tgl_pemerahan'].dt.hour
        
        enc_map = self.params['categorical_encodings']
        for col in ['jenis_sapi', 'jenis_pakan', 'kondisi_sapi', 'status_reproduksi']:
            if col in cow_data.columns and col in enc_map:
                cow_data[f'{col}_enc'] = cow_data[col].astype(str).str.lower().str.strip().map(enc_map[col]).fillna(0)
            elif col not in cow_data.columns and col in enc_map:
                cow_data[f'{col}_enc'] = 0

        feature_cols = self.params['feature_names']
        try:
            last_features = cow_data[feature_cols].tail(window_size).values.astype(np.float32)
        except KeyError as e:
            messagebox.showerror("Error Kolom", f"Kolom tidak cocok dengan model: {e}")
            return

        last_date = cow_data['tgl_pemerahan'].max()

        # 3. Scaling Input
        x_min = np.array(self.params['X_min'])
        x_max = np.array(self.params['X_max'])
        current_window = (last_features - x_min) / (x_max - x_min + 1e-7)

        predictions = []
        total_milk = 0
        try:
            days = int(self.days_entry.get())
        except:
            days = 7

        idx_bulan = feature_cols.index('bulan') if 'bulan' in feature_cols else -1
        idx_hari = feature_cols.index('hari_minggu') if 'hari_minggu' in feature_cols else -1

        # 4. Recursive Loop Prediksi (Dengan Time-Shifting Fix)
        for _ in range(days):
            input_name = self.ort_session.get_inputs()[0].name
            input_data = current_window.reshape(1, window_size, len(feature_cols)).astype(np.float32)
            
            pred_scaled = self.ort_session.run(None, {input_name: input_data})[0][0][0]

            y_min, y_max = self.params['y_min'], self.params['y_max']
            pred_liter = max(0, pred_scaled * (y_max - y_min) + y_min)
            
            next_date = last_date + timedelta(days=1)
            predictions.append((next_date.strftime("%Y-%m-%d"), pred_liter))
            total_milk += pred_liter
            
            new_row = current_window[-1].copy()
            new_row[0] = pred_scaled 
            
            if idx_bulan != -1:
                new_row[idx_bulan] = (next_date.month - x_min[idx_bulan]) / (x_max[idx_bulan] - x_min[idx_bulan] + 1e-7)
            if idx_hari != -1:
                new_row[idx_hari] = (next_date.weekday() - x_min[idx_hari]) / (x_max[idx_hari] - x_min[idx_hari] + 1e-7)

            current_window = np.roll(current_window, -1, axis=0)
            current_window[-1] = new_row
            last_date = next_date

        self.display_results(predictions, total_milk)

    def display_results(self, preds, total):
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Membaca data historis dari FILE CSV ✅\n")
        self.result_text.insert(tk.END, f"{'Tanggal':<12} | {'Prediksi (L)':<12}\n" + "-"*30 + "\n")
        for date_str, p in preds:
            self.result_text.insert(tk.END, f"{date_str:<12} | {p:>10.2f} L\n")
        self.result_text.insert(tk.END, "-"*30 + "\n")
        self.result_text.insert(tk.END, f"TOTAL {len(preds)} HARI: {total:.2f} Liter")
        self.result_text.config(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = CSVForecaster(root)
    root.mainloop()