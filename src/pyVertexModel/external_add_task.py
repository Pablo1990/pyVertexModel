import asyncio
import tkinter as tk
from tkinter import messagebox

import numpy as np
from src.pyVertexModel import task_manager

def show_alert(message):
    """
    Show an alert with an OK button.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    messagebox.showinfo("Alert", message)
    root.destroy()

async def main():
    """
    Add new routines to the task manager.
    """
    for i in range(10):
        print(f"Adding task with ID {i}")
        await task_manager.add_task(i)
        show_alert(f"Task with ID {i} added. Click OK to continue.")

async def run_all():
    await asyncio.gather(
        task_manager.start_task_manager(),
        main()
    )

if __name__ == "__main__":
    asyncio.run(run_all())