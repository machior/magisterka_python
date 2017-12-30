# from features.ExtractFeatures import extract_features
from tkinter.filedialog import askopenfilename

from features.GroupSigns import get_files
import tkinter as tk
from tkinter import filedialog


class SelectionWindow:
    out_path = None

    def __init__(self, in_path):
        self.in_path = in_path
        root = tk.Tk()

        self.in_path_string = tk.StringVar(value=in_path)
        self.out_path_string = tk.StringVar()
        tk.Button(root, text="Source Directory", command=self.get_source_dir).pack(side=tk.TOP)
        tk.Label(None, textvariable=self.in_path_string, fg='black').pack()
        tk.Button(root, text="Destination Directory", command=self.get_destination_dir).pack(side=tk.TOP)
        tk.Label(None, textvariable=self.out_path_string, fg='black').pack()

        tk.Button(root, text="Start", command=self.get_files).pack(side=tk.TOP)
        root.mainloop()

    def get_source_dir(self):
        self.in_path = filedialog.askdirectory(initialdir=self.in_path)
        self.in_path_string.set(self.in_path)

    def get_destination_dir(self):
        self.out_path = filedialog.askdirectory(initialdir=self.out_path)
        self.out_path_string.set(self.out_path)

    def get_files(self):
        if self.in_path is not None and self.out_path is not None:
            get_files(file_name='all', in_dir_path=self.in_path, out_dir_path=self.out_path)


if __name__ == '__main__':
    dir_path = '/media/bartek/120887D50887B5EF/POLITECHNIKA/Magisterka/SUSigP/Data/BlindSubCorpus/FORGERY'

    SelectionWindow(dir_path)

    # extract_features(files_urls=[
    #     '/media/bartek/120887D50887B5EF/POLITECHNIKA/Magisterka/SUSig/VisualSubCorpus/GENUINE/SESSION2/001_2_1.sig'])
