from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont


def save_pdf(text, path):

    c = canvas.Canvas(str(path))
    font_name = "STSong-Light"
    font_size = 12
    line_height = 18
    max_chars = 45

    pdfmetrics.registerFont(UnicodeCIDFont(font_name))
    c.setFont(font_name, font_size)

    y = 800

    for line in text.split("\n"):

        chunks = [line[i:i + max_chars] for i in range(0, len(line), max_chars)] or [""]

        for chunk in chunks:
            c.drawString(40, y, chunk)
            y -= line_height

            if y < 40:
                c.showPage()
                c.setFont(font_name, font_size)
                y = 800

    c.save()
