import os

import cv2
from PIL import Image 
import img2pdf
import pdf2image
# PyPDF2
from PyPDF2 import PdfFileWriter, PdfFileReader
# pdfminer
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine
from pdfminer.pdfpage import PDFPage


def get_txt_from_pdf(page):
    rsrcmgr = PDFResourceManager()
    laparams = LAParams(char_margin=0.5, line_margin=0.1)  # for 등기부등본
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    textboxes = []
    interpreter.process_page(page)
    layout = device.get_result()
    for lobj in layout:
        if isinstance(lobj, LTTextBox):
            x0, y0_orig, x1, y1_orig = lobj.bbox
            y0 = page.mediabox[3] - y1_orig
            y1 = page.mediabox[3] - y0_orig
            lobj.bbox = (x0, y0, x1, y1)
            textboxes.append(lobj)

    # reorder textboxes using box y0, x0 instead of box index
    textboxes.sort(key=lambda box: (-box.y0, box.x0))
    device.close()
    return textboxes


def crop_pdf(pdf_path, left, top, right, bottom):
    """
    text information remains
    """
    pdf = PdfFileReader(open(pdf_path, 'rb'))
    out = PdfFileWriter()
    page.mediaBox.upperRight = (page.mediaBox.getUpperRight_x() - right, page.mediaBox.getUpperRight_y() - top)
    page.mediaBox.lowerLeft  = (page.mediaBox.getLowerLeft_x()  + left,  page.mediaBox.getLowerLeft_y()  + bottom)
    out.addPage(page)    
    ous = open(target, 'wb')
    out.write(ous)
    ous.close()


def get_page_from_pdf(pdf_path):
    fp = open(pdf_path, 'rb')
    pages = [page for page in PDFPage.get_pages(fp)]
    return pages


def _img2pdf(img_path, pdf_path):
    image = Image.open(img_path) 
    pdf_bytes = img2pdf.convert(image.filename) 
    f = open(pdf_path, "wb")   
    f.write(pdf_bytes) 
    f.close() 


def _pdf2img(pdf_path):
    pages = pdf2image.convert_from_path(path, dpi=72)
    assert len(pages) == 1
    for page in pages:
        path = pdf_path.replace("pdf", "jpg")
        page.save(path, 'JPEG')
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_list.append(image)