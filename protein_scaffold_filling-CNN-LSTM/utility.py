
import os
from typing import Optional
from copy import deepcopy
from termcolor import colored

import fitz
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph

# registering a external font in python
pdfmetrics.registerFont(
    TTFont('mymonospace', 'fonts/AnonymousPro-Bold.ttf')
)

myParagraphStyle = ParagraphStyle(
    name='Normal',
    fontName='mymonospace',
    fontSize=12,
)

# =============================================================================
def mkdir_if_not_exists(path):
    try:
        os.mkdir(os.path.join(path))
    except FileExistsError:
        pass

# =============================================================================
def get_sequences(fasta_file):
    sequences = []
    lines = []
    with open(fasta_file, "r") as input_file:
        lines = list(filter(None, input_file.read().split("\n")))

    parts = []
    for line in lines:
        if line.startswith(">"):
            if parts:
                sequences.append("".join(parts))
            parts = []
        else:
            parts.append(line)
    if parts:
        sequences.append("".join(parts))
    return sequences

# =============================================================================
def snake_case_prettify(s):
    return " ".join(w.capitalize() for w in s.split("_"))

# =============================================================================
def print_sequence(seq, 
                   header: str=None, 
                   incorrect_indices: Optional[np.array]=None, 
                   correct_indices: Optional[np.array]=None):

    def highlight_indices(seq: np.array, indices: np.array, color: str):
        # We use deepcopy to prevent mutation and we cast to object so that we can treat contents as python strings
        # otherwise, it gets messed up as it treats each element as a single character
        newseq = deepcopy(seq).astype('object')
        newseq[indices] = np.vectorize(lambda x: colored(x, color, attrs=["bold"]))(seq[indices])
        return newseq

    newseq = deepcopy(seq)
    if correct_indices is not None and correct_indices.size != 0:
        newseq = highlight_indices(newseq, correct_indices, "green")
    if incorrect_indices is not None and incorrect_indices.size != 0:
        newseq = highlight_indices(newseq, incorrect_indices, "red")

    line_length = 40
    if header:
        print(header)
    print("=" * line_length)

    i = 0
    while i < len(newseq):
        print(" ".join(newseq[i: i+line_length]))
        i += line_length

# =============================================================================
def write_protein_scaffold_image(scaffold: np.array, incorrect_indices: np.array, correct_indices: np.array, filename: str):

    def add_color(s, color):
        return f'<font color="{color}">{s}</font>'

    pdf_filename = f"{filename}.pdf"

    # Create the raw html for the paragraph
    scaffold = scaffold.astype(object)
    for i in correct_indices:
        scaffold[i] = add_color(scaffold[i], "green")

    for i in incorrect_indices:
        scaffold[i] = add_color(scaffold[i], "red")
        
    para = Paragraph("".join(elem for elem in scaffold), myParagraphStyle)

    # creating a pdf object
    canvasSize = (125*mm, 18*mm)  
    pdf = canvas.Canvas(pdf_filename)
    pdf.setTitle(pdf_filename)
    pdf.setPageSize(canvasSize)
    pdf.setFont('mymonospace', 18)

    para.wrapOn(pdf, canvasSize[0] + 1*mm, 500*mm)  # size of 'textbox' for linebreaks etc.
    para.drawOn(pdf, 0*mm, 1*mm)                    # position of text / where to draw
  
    # saving the pdf
    pdf.save()

    # It's stupid to save both the pdf and the png, but I don't have time to do it right
    dpi = 300  # choose desired dpi here
    zoom = dpi / 72  # zoom factor, standard: 72 dpi
    magnify = fitz.Matrix(zoom, zoom)  # magnifies in x, resp. y direction
    doc = fitz.open(pdf_filename)  # open document

    pix = doc[0].get_pixmap(matrix=magnify)  # render page to an image
    pix.save(f"{filename}.png")
