import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import firebase_admin
from firebase_admin import credentials, firestore
import csv
import datetime

class CowLSTMExporter:
    def __init__(self, root):
        self.root = root
        self.root.title("Cow Data Exporter (Quota Safe)")
        self.root.geometry("500x550") # Made window slightly taller

        # Variables
        self.json_key_path = tk.StringVar()
        self.csv_filename = tk.StringVar(value="dataset_sapi_lstm.csv")
        self.limit_cows = tk.IntVar(value=5) # Default limit to 5 cows for safety
        self.status_var = tk.StringVar(value="Ready")

        self.create_widgets()

    def create_widgets(self):
        # 1. Auth
        ttk.Label(self.root, text="1. Service Account Key", font=("Arial", 10, "bold")).pack(pady=5)
        frame_auth = ttk.Frame(self.root)
        frame_auth.pack(pady=5)
        ttk.Entry(frame_auth, textvariable=self.json_key_path, width=40).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_auth, text="Browse", command=self.browse_file).pack(side=tk.LEFT)

        # 2. Output
        ttk.Label(self.root, text="2. Output Filename", font=("Arial", 10, "bold")).pack(pady=10)
        ttk.Entry(self.root, textvariable=self.csv_filename, width=40).pack(pady=5)

        # 3. Safety Limits (NEW)
        ttk.Label(self.root, text="3. Safety Limit (Quota Control)", font=("Arial", 10, "bold")).pack(pady=10)
        
        frame_limit = ttk.Frame(self.root)
        frame_limit.pack(pady=5)
        ttk.Label(frame_limit, text="Max Cows to Process:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(frame_limit, textvariable=self.limit_cows, width=10).pack(side=tk.LEFT, padx=5)
        
        # Helper note
        lbl_note = tk.Label(self.root, text="(Set to 0 for UNLIMITED - Be careful!)", 
                            font=("Arial", 8), fg="red")
        lbl_note.pack(pady=0)

        # 4. Info
        info_text = (
            "Logic: This script will fetch X cows, and for each cow,\n"
            "fetch ALL their milking records.\n"
            "Quota Used ≈ (Cows Limit) + (Total Milking Records)"
        )
        tk.Label(self.root, text=info_text, fg="gray", justify=tk.LEFT, bg="#f0f0f0", padx=10, pady=5).pack(pady=15)

        # 5. Button
        btn_export = tk.Button(self.root, text="START EXPORT", bg="#4CAF50", fg="white", 
                               font=("Arial", 12, "bold"), command=self.run_deep_export)
        btn_export.pack(pady=10, ipadx=20, ipady=10)

        # Status
        ttk.Label(self.root, textvariable=self.status_var, foreground="blue").pack(pady=5)

    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if filename:
            self.json_key_path.set(filename)

    def run_deep_export(self):
        key_path = self.json_key_path.get()
        limit_val = self.limit_cows.get()

        if not key_path:
            messagebox.showerror("Error", "Please select JSON key first")
            return

        try:
            self.status_var.set("Connecting to Firestore...")
            self.root.update()

            # Init Firebase
            try:
                app = firebase_admin.get_app()
            except ValueError:
                cred = credentials.Certificate(key_path)
                firebase_admin.initialize_app(cred)
            
            db = firestore.client()

            # --- THE NESTED EXPORT LOGIC ---
            self.status_var.set("Fetching Parent Data (DataSapi)...")
            self.root.update()

            # 1. Get Cows (WITH LIMIT)
            cows_ref = db.collection('DataSapi')
            
            if limit_val > 0:
                # This limits how many PARENT docs we read
                cows_query = cows_ref.limit(limit_val)
                print(f"Limiting export to {limit_val} cows.")
            else:
                cows_query = cows_ref

            cows = list(cows_query.stream())
            
            if not cows:
                self.status_var.set("No cows found.")
                messagebox.showwarning("Empty", "No data found in 'DataSapi'")
                return

            total_records = 0
            flattened_data = []

            # 2. Loop through every Cow
            for i, cow in enumerate(cows):
                cow_data = cow.to_dict()
                cow_id = cow.id
                
                self.status_var.set(f"Processing Cow {i+1}/{len(cows)}: {cow_data.get('Nama', cow_id)}")
                self.root.update()

                # 3. Get Subcollection 'Pemerahan' for THIS cow
                # We usually want ALL history for LSTM, so we don't limit here
                milking_ref = cows_ref.document(cow_id).collection('Pemerahan')
                milking_events = milking_ref.stream()

                for event in milking_events:
                    event_data = event.to_dict()
                    
                    # --- MERGE LOGIC ---
                    row = {
                        'sapi_id': cow_id,
                        'nama_sapi': cow_data.get('Nama'),
                        'jenis_sapi': cow_data.get('Jenis'),
                        'tgl_lahir': self.format_date(cow_data.get('TglLahir')),
                        
                        'pemerahan_id': event.id,
                        'tgl_pemerahan': self.format_date(event_data.get('tglPemerahan')),
                        
                        'jumlah_susu': event_data.get('jumlahSusu'),
                        'volume_pakan': event_data.get('volumePakan'),
                        'jenis_pakan': event_data.get('jenisPakan'),
                        'kondisi_sapi': event_data.get('kondisiSapi'),
                        'pemerah': event_data.get('pemerah'),
                        'status_reproduksi': event_data.get('statusReproduksi'),
                    }
                    flattened_data.append(row)
                    total_records += 1

            # 4. Save to CSV
            if flattened_data:
                # Sort the data for LSTM (by Cow, then by Date)
                # Note: This is a simple string sort, for perfect date sorting use Pandas later
                flattened_data.sort(key=lambda x: (x['sapi_id'], x['tgl_pemerahan']))

                keys = flattened_data[0].keys()
                with open(self.csv_filename.get(), 'w', newline='', encoding='utf-8') as f:
                    dict_writer = csv.DictWriter(f, fieldnames=keys)
                    dict_writer.writeheader()
                    dict_writer.writerows(flattened_data)
                
                self.status_var.set(f"Done! Exported {total_records} rows.")
                messagebox.showinfo("Success", f"Exported {total_records} rows from {len(cows)} cows.")
            else:
                self.status_var.set("No milking data found.")
                messagebox.showwarning("Empty", "Cows found, but they had no 'Pemerahan' records.")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Error occurred")
            print(e)

    def format_date(self, val):
        """Helper to convert Firestore Timestamps to clean String"""
        if isinstance(val, datetime.datetime):
            return val.strftime("%Y-%m-%d %H:%M:%S")
        return str(val)

if __name__ == "__main__":
    root = tk.Tk()
    app = CowLSTMExporter(root)
    root.mainloop()