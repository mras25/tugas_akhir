# gunicorn.conf.py
# Konfigurasi untuk meminimalkan penggunaan RAM di Render free tier (512MB)

workers = 1          # Jangan lebih dari 1 — setiap worker load TensorFlow ~300MB
threads = 2          # Gunakan thread daripada worker tambahan
timeout = 300        # Beri waktu lebih untuk prediksi model (default 30s terlalu singkat)
worker_class = "gthread"
preload_app = True   # Load model sekali saja di master process, tidak per-worker
max_requests = 50    # Restart worker setelah 50 request untuk cegah memory leak
max_requests_jitter = 10
