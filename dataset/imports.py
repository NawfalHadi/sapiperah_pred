import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
import threading

class FirestoreUploaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Firestore Subcollection Uploader")
        self.root.geometry("450x300")
        self.root.resizable(False, False)

        self.db = None
        self.csv_filepath = None

        # 1. Key Selection
        tk.Label(root, text="1. Load Firebase Service Account Key (.json):").pack(pady=(10, 0))
        self.btn_key = tk.Button(root, text="Browse Key", command=self.load_key)
        self.btn_key.pack()
        self.lbl_key_status = tk.Label(root, text="Not loaded", fg="red")
        self.lbl_key_status.pack()

        # 2. CSV Selection
        tk.Label(root, text="2. Select Dataset (.csv):").pack(pady=(10, 0))
        self.btn_csv = tk.Button(root, text="Browse CSV", command=self.load_csv, state=tk.DISABLED)
        self.btn_csv.pack()
        self.lbl_csv_status = tk.Label(root, text="Not selected", fg="red")
        self.lbl_csv_status.pack()

        # 3. Max Data Limit
        tk.Label(root, text="3. Max Data to Upload (Kosongkan untuk upload semua):").pack(pady=(10, 0))
        self.entry_max = tk.Entry(root, width=15, justify='center')
        self.entry_max.pack()

        # Progress & Upload Button
        self.progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress.pack(pady=15)

        self.btn_upload = tk.Button(root, text="Upload to DataBackup", command=self.start_upload, bg="green", fg="white", state=tk.DISABLED)
        self.btn_upload.pack()

    def load_key(self):
        key_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if key_path:
            try:
                if not firebase_admin._apps:
                    cred = credentials.Certificate(key_path)
                    firebase_admin.initialize_app(cred)
                self.db = firestore.client()
                
                self.lbl_key_status.config(text="Key loaded successfully!", fg="green")
                self.btn_csv.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to initialize Firebase:\n{str(e)}")

    def load_csv(self):
        self.csv_filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if self.csv_filepath:
            self.lbl_csv_status.config(text=f"Selected: {self.csv_filepath.split('/')[-1]}", fg="green")
            self.btn_upload.config(state=tk.NORMAL)

    def start_upload(self):
        self.btn_key.config(state=tk.DISABLED)
        self.btn_csv.config(state=tk.DISABLED)
        self.btn_upload.config(state=tk.DISABLED)
        self.entry_max.config(state=tk.DISABLED)
        
        threading.Thread(target=self.process_upload, daemon=True).start()

    def process_upload(self):
        try:
            # Membaca data CSV
            df = pd.read_csv(self.csv_filepath)
            
            # Membatasi data jika input Max Data diisi
            max_limit_str = self.entry_max.get().strip()
            if max_limit_str.isdigit():
                max_limit = int(max_limit_str)
                df = df.head(max_limit)
            
            total_records = len(df)
            
            if total_records == 0:
                self.root.after(0, lambda: messagebox.showwarning("Warning", "Tidak ada data untuk diupload!"))
                self.root.after(0, self.reset_buttons)
                return

            self.progress['maximum'] = total_records
            self.progress['value'] = 0

            batch = self.db.batch()
            operations_count = 0
            records_processed = 0
            uploaded_sapi_ids = set()

            for index, row in df.iterrows():
                sapi_id = str(row['sapi_id'])
                pemerahan_id = str(row['pemerahan_id'])
                
                sapi_ref = self.db.collection('DataBackup').document(sapi_id)
                pemerahan_ref = sapi_ref.collection('pemerahan').document(pemerahan_id)

                if sapi_id not in uploaded_sapi_ids:
                    sapi_data = {
                        'Nama': row.get('nama_sapi', ''),
                        'Jenis': row.get('jenis_sapi', ''),
                        'TglLahir': row.get('tgl_lahir', ''),
                        'Kelamin': 'Betina'
                    }
                    batch.set(sapi_ref, sapi_data, merge=True)
                    uploaded_sapi_ids.add(sapi_id)
                    operations_count += 1

                pemerahan_data = {
                    'jenisPakan': row.get('jenis_pakan', ''),
                    'jumlahSusu': row.get('jumlah_susu', 0),
                    'kondisiSapi': row.get('kondisi_sapi', ''),
                    'pemerah': row.get('pemerah', ''),
                    'statusReproduksi': row.get('status_reproduksi', ''),
                    'tglPemerahan': row.get('tgl_pemerahan', ''),
                    'volumePakan': row.get('volume_pakan', 0)
                }
                batch.set(pemerahan_ref, pemerahan_data)
                operations_count += 1
                records_processed += 1

                if operations_count >= 490:
                    batch.commit()
                    batch = self.db.batch()
                    operations_count = 0
                
                self.root.after(0, self.update_progress, records_processed)

            if operations_count > 0:
                batch.commit()

            # --- BAGIAN BARU: PRINT SUMMARY PER SAPI ---
            print("\n" + "="*45)
            print("LAPORAN DATA YANG BERHASIL DI-UPLOAD")
            print("="*45)
            
            # Kelompokkan berdasarkan nama_sapi dan status_reproduksi
            grouped = df.groupby('nama_sapi')['status_reproduksi'].value_counts().unstack(fill_value=0)
            
            for sapi_name, counts in grouped.iterrows():
                hamil_count = counts.get('Hamil', 0)
                laktasi_count = counts.get('Laktasi', 0)
                print(f"{sapi_name} : hamil {hamil_count} data | laktasi {laktasi_count} data")
            
            print("="*45 + "\n")
            # -------------------------------------------

            self.root.after(0, lambda: self.upload_complete(total_records, len(uploaded_sapi_ids)))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Upload Error", str(e)))
            self.root.after(0, self.reset_buttons)

    def update_progress(self, value):
        self.progress['value'] = value

    def upload_complete(self, total_uploaded, total_sapi):
        messagebox.showinfo(
            "Success", 
            f"Berhasil!\n\n"
            f"- Data Pemerahan diunggah: {total_uploaded}\n"
            f"- Total Sapi (Dokumen Induk) terpengaruh: {total_sapi}\n\n"
            f"(Cek terminal/console untuk melihat rincian per sapi)"
        )
        self.reset_buttons()
        self.progress['value'] = 0

    def reset_buttons(self):
        self.btn_key.config(state=tk.NORMAL)
        self.btn_csv.config(state=tk.NORMAL)
        self.btn_upload.config(state=tk.NORMAL)
        self.entry_max.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = FirestoreUploaderApp(root)
    root.mainloop()