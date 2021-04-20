import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd

import pdf2image
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine
from pdfminer.pdfpage import PDFPage


def reorganize_table(page_ind, img_cell_coords, pdf_textboxes, row_num):
    max_column = max(img_cell_coords, key=len)
    df = pd.DataFrame(columns=range(len(max_column)), index=range(row_num))
    for r_ind, row_cells in enumerate(img_cell_coords):
        for c_ind, cell in enumerate(row_cells):
            for pdf_box in pdf_textboxes:
                x0, y0, x1, y1 = pdf_box.bbox
                if x0>=cell[0] and y0>=cell[1] and x1<=cell[2] and y1<=cell[3]:
                    content = pdf_box.get_text().rstrip().replace(u'\xa0', u' ')
                    if type(df.loc[r_ind][c_ind]) == float: # nan
                        df.loc[r_ind][c_ind] = content
                    else:
                        df.loc[r_ind][c_ind] = df.loc[r_ind][c_ind]+'\n'+content
    df.to_csv(args.result_path + '/report-register-complex'+str(page_ind)+'.csv', 
                                        encoding='utf-8-sig', header=False)


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


def get_cropped_image(image, x, y, w, h):
    cropped_image = image[ y:y+h , x:x+w ]
    return cropped_image


def get_ROI(image, horizontal, vertical, left_line_index, right_line_index, top_line_index, bottom_line_index, offset=5):
    '''
    horizontal : [start_x, y, end_x, y]
    vertical : [x, end_y, x, start_y]
    '''
    x1 = vertical[left_line_index][2] + offset
    y1 = horizontal[top_line_index][3] + offset
    x2 = vertical[right_line_index][2] - offset
    y2 = horizontal[bottom_line_index][3] - offset
    w = x2 - x1
    h = y2 - y1
    cropped_image = get_cropped_image(image, x1, y1, w, h)
    return cropped_image, (x1, y1, x2, y2)


def get_register_v_cnts(first_row, last_row, post_horizontal_cnts, post_vertical_cnts):
    columns = []
    min_y = post_horizontal_cnts[first_row][1]
    max_y = post_horizontal_cnts[last_row][1]
    for v in post_vertical_cnts:
        # start_y of vertical contour <= max_y && end_y of vertical contour >= min_y
        if v[3] < max_y and v[1] > min_y:
            columns.append(v)
    columns.sort(key=lambda x:x[0])
    return columns


def get_register_cell(img, first_row, last_row, post_horizontal_cnts, post_vertical_cnts, save=False):
    '''
    can be applied for only report-register document(등기부등본)
    - the sheet should have same column format
    '''
    # find the coordinates of each cell in ROI
    coord_cells = [[] for _ in range(first_row, last_row)]
    img_cells = []
    for i in range(first_row, last_row):
        column_cnts = get_register_v_cnts(i, i+1, post_horizontal_cnts, post_vertical_cnts) # get vertical lines in ROI and sort
        column_num = len(column_cnts) - 1
        for j in range(column_num):
            # set index of cell contour line
            left_line_index = j
            right_line_index = j+1
            top_line_index = i
            bottom_line_index = i+1
            # get cell from table
            crop_cell, coord = get_ROI(img, post_horizontal_cnts, column_cnts, left_line_index, right_line_index, top_line_index, bottom_line_index, offset=0)
            coord_cells[i].append(list(coord))
            img_cells.append(crop_cell)
    if save:
        for i in range(len(img_cells)):
            cv2.imwrite(args.pre_save_path+str(i)+'_crop_img.jpg', img_cells[i])
    return coord_cells, img_cells


def write_contour_txt(post_horizontal_cnts, post_vertical_cnts):
    f_h = open(args.pre_save_path+'/horizontal.txt', 'w')
    f_v = open(args.pre_save_path+'/vertical.txt', 'w')
    f_h.write(str(post_horizontal_cnts))
    f_v.write(str(post_vertical_cnts))
    f_h.close()
    f_v.close()


def draw_contour(image, page_ind, post_horizontal_cnts, post_vertical_cnts):
    img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) # to save contour with color
    for c in post_horizontal_cnts:
        cv2.line(img_color, (c[0],c[1]), (c[2], c[3]), (0,255,0), 5)
    for c in post_vertical_cnts:
        cv2.line(img_color, (c[0],c[3]), (c[2], c[1]), (0,0,255), 5)
    cv2.imwrite(args.pre_save_path+'/filter_contour_page'+str(page_ind)+'.jpg', img_color)


def filter_vertical_lines(h_lns, v_lns):
    # remove vertical contours that are not part of table
    hor_y_cand = np.array(h_lns)[:, 1] 
    filter_vertical = []
    for line in v_lns:
        # remove veritcal line that are not matched with any horizontal line 
        if line[1] in hor_y_cand or line[3] in hor_y_cand:
            # [토지대장]check if x of vertical line is included in the x-scope of horizontal line 
            point_lst = np.concatenate((np.where(hor_y_cand == line[1])[0], np.where(hor_y_cand == line[3])[0]))
            flag = False
            for p in point_lst:
                if line[0] in range(h_lns[p][0]-2, h_lns[p][2]+2):
                    flag = True
                    break
            if flag:
                filter_vertical.append(line)
    return filter_vertical


def filter_horizontal_lines(h_lns, v_lns):
    ver_x_cand = np.array(v_lns)[:, 0]
    filter_horizontal = []
    for line in h_lns:
        # [등기부등본] 폐기사항 line인데 vertical contour 만나는 경우 있음 
        # table의 시작점에서 (table 길이의 1/4지점)보다 떨어진 h line 제거
        if line[0] > min(ver_x_cand) + (max(ver_x_cand)-min(ver_x_cand))/4:
            continue
        # remove horizontal line that are not matched with any vertical line 
        if line[0] in ver_x_cand or line[2] in ver_x_cand:
            filter_horizontal.append(line)
    return filter_horizontal


def postprocess_cnts(horizontal_cnts, vertical_cnts):
    '''
    transform contour coordinates from rectangle format to line format 
    - findContour() result format(rectangle) 
        [[left_up_x, y], [left_down_x, y], [right_down_x, y], [right_up_x, y]]
    - line format
        [startpoint_x, startpoint_y, endpoint_x, endpoint_y]
    '''
    # find unique edge coordinates of each contour
    post_horizontal = []
    post_vertical = []
    vertical_y_cand = set()
    horizontal_x_cand = set()

    # postprocess horizontal contours
    for i in range(len(horizontal_cnts)):
        coords = np.array(horizontal_cnts[i]).squeeze()
        coords_x = coords[:, 0]  # (rectangle을 이루는 좌표 개수, x좌표)
        start_x = int(min(coords_x))
        end_x = int(max(coords_x))
        horizontal_x_cand.update([start_x, end_x])
        post_horizontal.append([start_x, 0, end_x, 0])  #initialize y as 0

    # postprocess vertical contours
    for i in range(len(vertical_cnts)):
        coords = np.array(vertical_cnts[i]).squeeze()
        coords_y = coords[:, 1]
        start_y = int(min(coords_y))
        end_y = int(max(coords_y))
        vertical_y_cand.update([start_y, end_y])
        post_vertical.append([0, end_y, 0, start_y])  #initialize x as 0

    # match horizontal contour y with vertical contours y to filter vertical line
    for i in range(len(horizontal_cnts)):
        coords = np.array(horizontal_cnts[i])
        coords_y = coords.squeeze()[:, 1] 
        for h_y in set(coords_y):
            y = min(vertical_y_cand, key=lambda v_y : abs(v_y-h_y))
            diff = abs(h_y - y)
            if diff <= 3:
                post_horizontal[i][1] = y
                post_horizontal[i][3] = y
            else:
                post_horizontal[i][1] = np.median(coords_y).astype(int)
                post_horizontal[i][3] = np.median(coords_y).astype(int)
    
    # match vertical contour x with horizontal contours x to filter horizontal line
    for i in range(len(vertical_cnts)):
        coords = np.array(vertical_cnts[i])
        coords_x = coords.squeeze()[:, 0] 
        for v_x in set(coords_x):
            x = min(horizontal_x_cand, key=lambda h_x : abs(h_x-v_x))
            diff = abs(v_x - x)
            if diff <= 3:
                post_vertical[i][0] = x
                post_vertical[i][2] = x
            else:
                post_vertical[i][0] = np.median(coords_x).astype(int)
                post_vertical[i][2] = np.median(coords_x).astype(int)

    # reverse order from bottom-right to up-left
    post_horizontal.reverse()
    post_vertical.reverse()
    return post_horizontal, post_vertical


def detect_contour(image):
    '''
    find the contours of the table in the image
    return: horizontal contours, vertical_contours
    '''
    # detect horizontal lines
    inv_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
    detect_horizontal = cv2.morphologyEx(inv_img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    h_cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_cnts = h_cnts[0] if len(h_cnts) == 2 else h_cnts[1]

    # detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,15))
    detect_vertical = cv2.morphologyEx(inv_img, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    v_cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_cnts = v_cnts[0] if len(v_cnts) == 2 else v_cnts[1]
    return h_cnts, v_cnts


def _im_read(args):
    # IMREAD_COLOR, IMREAD_UNCHANGED
    path = args.path
    img_list = []
    if path[-3:] == "jpg":
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_list.append(image)
    elif path[-3:] == "pdf":
        # convert to image
        pages = pdf2image.convert_from_path(path, dpi=72)
        for page in pages:
            path = path.replace("pdf", "jpg")
            page.save(path, 'JPEG')
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_list.append(image)
    else:
        raise "unsupported input type"
    return img_list

def get_page_from_pdf(args):
    fp = open(args.path, 'rb')
    pages = [page for page in PDFPage.get_pages(fp)]
    return pages


def main(args):

    # read image as gray scale
    pages = get_page_from_pdf(args)
    images = _im_read(args)

    start_time = time.time()
    for page_indx, img in enumerate(images):

        # Detect table contours
        horizontal_cnts, vertical_cnts = detect_contour(img)

        # Postprocess detected contours
        post_h_cnts, post_v_cnts = postprocess_cnts(horizontal_cnts, vertical_cnts)

        # no table in page
        if len(post_h_cnts) == 0 or len(post_v_cnts) == 0:
            continue

        # Filter contours not part of table
        post_v_cnts = filter_vertical_lines(post_h_cnts, post_v_cnts)
        post_h_cnts = filter_horizontal_lines(post_h_cnts, post_v_cnts)
        draw_contour(img, page_indx, post_h_cnts, post_v_cnts)
        # write_contour_txt(post_h_cnts, post_v_cnts)

        # Get ROI coordinate of each cell in page
        first_line_index = 0
        last_line_index = len(post_h_cnts)-1 
        coord_cells, _ = get_register_cell(img, first_line_index, last_line_index, post_h_cnts, post_v_cnts, save=True)

        # Find the textboxes in pdf
        pdf_textboxes = get_txt_from_pdf(pages[page_indx])

        # Assign textboxes of pdf in each cell of image
        reorganize_table(page_indx, coord_cells, pdf_textboxes, last_line_index)

    print("Total time elapsed: {:.4f} s".format(time.time()-start_time))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ocr")
    parser.add_argument(
        "--path",
        type=str
    )
    parser.add_argument(
        "--pre_save_path",
        type=str
    )
    parser.add_argument(
        "--result_path",
        type=str
    )

    global args
    args = parser.parse_args()

    path = args.path
    img_name = os.path.basename(path)[:-4]
    pre_save_path ='./preprocess/'
    if not os.path.exists(pre_save_path):
        os.mkdir(pre_save_path)
    result_path ='./results/'
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    args.pre_save_path = pre_save_path
    args.result_path = result_path

    main(args)

