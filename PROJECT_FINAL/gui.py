import tkinter as tk
from tkinter import messagebox
import subprocess
import os
import signal

# GLOBAL PROCESS
capture_process = None

# START CAPTURE
def start_capture():
    global capture_process

    if capture_process is not None:
        messagebox.showinfo(
            "Info",
            "Le système tourne déjà."
        )
        return

    try:
        capture_process = subprocess.Popen([
            "/home/alice/PROJET_FINAL/tf-env39/bin/python",
            "musicam.py"
            ])
        status_label.config(
            text="Capture active",
            fg="green")

    except Exception as e:
        messagebox.showerror("Erreur",str(e))

# STOP CAPTURE

def stop_capture():
    global capture_process
    if capture_process is None:
        return
    try:
        if os.name == 'nt':
            capture_process.terminate()
        else:
            os.kill(capture_process.pid,signal.SIGTERM)

        capture_process = None

        status_label.config(
            text="Capture arrêtée",
            fg="red"
        )

    except Exception as e:
        messagebox.showerror("Erreur",str(e))

# RENDER VIDEO
def render_video():
    try:
        status_label.config(
            text="Rendu vidéo...",
            fg="orange")

        root.update()

        subprocess.run(
            ["python", "video.py"],
            check=True)

        status_label.config(
            text="Vidéo générée",
            fg="blue")

        messagebox.showinfo(
            "Terminé",
            "La vidéo a été générée.")

    except Exception as e:
        messagebox.showerror("Erreur",str(e))

# OPEN VIDEO
def open_video():
    video_path = "motion_layers.mp4"
    if not os.path.exists(video_path):
        messagebox.showwarning(
            "Vidéo absente",
            "Aucune vidéo trouvée."
        )
        return
    try:
        if os.name == 'nt':
            os.startfile(video_path)
        else:
            subprocess.Popen(["xdg-open",video_path])
    except Exception as e:
        messagebox.showerror("Erreur",str(e))

# WINDOW
root = tk.Tk()

root.title("Motion Music System")
root.geometry("420x350")
root.configure(bg="#111111")

# TITLE
label_title = tk.Label(
    root,
    text="Interactive Motion Music",
    font=("Arial", 18, "bold"),
    bg="#111111",
    fg="white"
)

label_title.pack(pady=20)

# STATUS
status_label = tk.Label(
    root,
    text="Système arrêté",
    font=("Arial", 12),
    bg="#111111",
    fg="red"
)

status_label.pack(pady=10)

# CLEAN EXIT
def on_close():

    stop_capture()

    root.destroy()

# BUTTONS
button_style = {
    "font": ("Arial", 12, "bold"),
    "width": 22,
    "height": 2,
    "bd": 0
}

start_button = tk.Button(
    root,
    text="Lancer Capture",
    command=start_capture,
    bg="#2ecc71",
    fg="white",
    **button_style
)

start_button.pack(pady=10)

stop_button = tk.Button(
    root,
    text="Arrêter Capture",
    command=stop_capture,
    bg="#e74c3c",
    fg="white",
    **button_style
)

stop_button.pack(pady=10)

render_button = tk.Button(
    root,
    text="Créer Vidéo Analyse",
    command=render_video,
    bg="#3498db",
    fg="white",
    **button_style
)

render_button.pack(pady=10)

open_button = tk.Button(
    root,
    text="Ouvrir Vidéo",
    command=open_video,
    bg="#9b59b6",
    fg="white",
    **button_style
)

open_button.pack(pady=10)

quit_button = tk.Button(
    root,
    text="Quitter Programme",
    command=on_close,
    bg="#555555",
    fg="white",
    **button_style
)

quit_button.pack(pady=10)

root.protocol(
    "WM_DELETE_WINDOW",
    on_close
)

# START GUI
root.mainloop()
