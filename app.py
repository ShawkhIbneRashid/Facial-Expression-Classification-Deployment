import os
from PIL import Image
import pytesseract
import sys
from pdf2image import convert_from_path
import os
import cv2
import json
import numpy as np
from json2html import *
import webbrowser
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename
import re


ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
UPLOAD_FOLDER = 'uploads'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
           
def rotate_bound(image, angle):
    """Rotate image with the given angle
    :param type image: input image
    :param type angle: Angle to be rotated
    :return: rotated image
    :rtype: numpy.ndarray
    """
    (h, w) = image.shape[:2]
    ### centroid
    (cX, cY) = (w // 2, h // 2)
    ### creating rotation matrix
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH))



def pdf_to_json(file):
    
        
    image=cv2.imread(file)
    newdata=pytesseract.image_to_osd(image)
    angle=re.search('(?<=Rotate: )\d+', newdata).group(0)
    print('osd angle:',angle)
    if angle=='90':
        skew_corrected_image=rotate_bound(image,float(angle))
    else:
        skew_corrected_image = image
        
    text_tmp = str(((pytesseract.image_to_string(skew_corrected_image, config='--psm 6'))))
    p = '[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+'
    count = 0
    if re.search(p, text_tmp) is not None:
        for catch in re.finditer(p, text_tmp):
            count+=1
    if count>30:
        text = str(((pytesseract.image_to_string(skew_corrected_image, config='--psm 6'))))
        text = text.replace('|', '')
        lst = text.split('\n')
        lst = lst[1:-1]
        title = lst[0] + lst[1]
        lst = lst[2:]
        a_dict = {}
        a_dict["Title"] = title
        count = 0
        for i in lst:
            if len(i)>3:
                count+=1
                key = "row "+str(count)
                a_dict[key] = i
        
        json_object = json.dumps(a_dict, indent = 4)
        #with open('num_count/'+PDF_file[0:-4] +'_' +filename[10:-4] + '.json', 'w') as f:
        #    json.dump(json_object, f)
        
        table_attr = "class=" + "\"table table-bordered table-hover\""
        table_html = json2html.convert(json=json_object, table_attributes=table_attr)
        
        title = "table" + str(i)
        fileName = title + ".html"
        file = open(fileName, 'w')
        
        html_open = """<html>
        <head><link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css"></head>
        <body><div>"""
        
        html_close = """</div></body>
        </html>"""
        
        # Create HTML file and add tables
        html = html_open + table_html + html_close
        
        file.write(html)
        file.close()
        
        # Change path to reflect file location
        filename =  'file:///C:/Users/100790606/notebook_files/dataset/user_interface/' + title + '.html'
        
        webbrowser.open_new_tab(filename)
            
            
    #for file in os.listdir('uploads'): 
    #    if file.endswith('.png'):
    #        os.remove('uploads/'+file) 
    
    output = "Tables are converted to JSON objects and showed as tables"
    return output

app = Flask(__name__, template_folder='Templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def template_test():
    return render_template('home.html', label='', imagesource='file://null')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            output = pdf_to_json(file_path)
    return render_template("home.html", label=output, imagesource=file_path)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    app.run(threaded=False)