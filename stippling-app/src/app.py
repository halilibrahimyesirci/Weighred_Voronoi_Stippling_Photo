import customtkinter as ctk
from gui.main_window import MainWindow

def main():
    ctk.set_appearance_mode("System")  # Modes: "System", "Dark", "Light"
    ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"

    app = MainWindow()
    app.mainloop()

if __name__ == "__main__":
    main()