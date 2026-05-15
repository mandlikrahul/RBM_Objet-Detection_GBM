import tkinter as tk
from tkinter import messagebox


def open_popup(message: str) -> str:
    """
    Show a Yes/No dialog and return the user's choice as 'Yes' or 'No'.

    Parameters
    ----------
    message : str
        The question displayed in the dialog title.

    Returns
    -------
    str
        'Yes' or 'No'
    """
    result = {"choice": "No"}

    root = tk.Tk()
    root.withdraw()  # hide the main blank window

    answer = messagebox.askyesno(title="Execution Mode", message=message)
    result["choice"] = "Yes" if answer else "No"

    root.destroy()
    return result["choice"]
