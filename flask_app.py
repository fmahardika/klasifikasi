from distutils import extension
from email import message
from flask import Flask, render_template
from flask import request
from flask import redirect, url_for
import os
from matplotlib.pyplot import angle_spectrum
import numpy as np
import pandas as pd
import scipy
import sklearn
import skimage
import skimage.color
import skimage.transform
import skimage.feature
import skimage.io
import pickle


app = Flask(__name__)


BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')
MODEL_PATH = os.path.join(BASE_PATH,'static/models/')

##------------------Load Model------------
model_knn_path = os.path.join(MODEL_PATH,'dsa_image_classification_knn_akurasi_93.pickle')
scaler_path = os.path.join(MODEL_PATH,'dsa_scaler_knn.pickle')
model_knn = pickle.load(open(model_knn_path,'rb'))
scaler = pickle.load(open(scaler_path,'rb'))

@app.errorhandler(404)
def error404(error):
    message = "ERROR 404 OCCURED. Page Not Found. Silahkan kembali ke homepage"
    return render_template("error.html",message=message) #page not found

@app.errorhandler(405)
def error405(error):
    message = "ERROR 405 OCCURED. Method Not Found"
    return render_template("error.html",message=message)

@app.errorhandler(500)
def error500(error):
    message = "INTERNAL ERROR 500. Error occurs in the program"
    return render_template("error.html",message=message)


@app.route('/',methods=['GET','POST'])
def index():
    if request.method == "POST":
        upload_file = request.files['image_name']
        filename = upload_file.filename
        print('File telah berhasil diupload = ',filename)
        # validasi ekstensi file (.jpg, .png, .jpeg)
        ext = filename.split('.')[-1]
        print('Ekstensi file =',ext)
        if ext.lower() in ['png','jpg','jpeg']:
            # simpan gambar
            path_save = os.path.join(UPLOAD_PATH,filename)
            upload_file.save(path_save)
            print('File berhasil disimpan')
            #send to pipeline Model
            results = pipeline_model(path_save,scaler,model_knn)
            hei = getheight(path_save)
            print(results)
            return render_template('upload.html',fileupload=True,extension=False,data=results, dt=list(results.keys)[0], image_filename=filename)

        else:
            print('File hanya dapat diproses dalam ekstensi .jpg, .png, .jpeg')
            return render_template('upload.html',extension=True,fileupload=False)

        
    else:
         return render_template('upload.html',fileupload=False,extension=False)

def getheight(path):
    img = skimage.io.imread(path)
    h,w,_ = img.shape
    ascept = h/w
    given_witdh = 200
    height = given_witdh*ascept
    return height



def pipeline_model(path,scaler_transform,model_knn):
    # pipeline model
    image = skimage.io.imread(path)
    # transform image into 100 x 100
    image_resize = skimage.transform.resize(image,(100,100))
    image_scale = 255*image_resize
    image_transform = image_scale.astype(np.uint8)
    # rgb to gray
    gray = skimage.color.rgb2gray(image_transform)
    # hog feature
    feature_vector = skimage.feature.hog(gray,
                                  orientations=9,
                                  pixels_per_cell=(8,8),cells_per_block=(3,3))
    # scaling
    
    scalex = scaler_transform.transform(feature_vector.reshape(1,-1))
    result = model_knn.predict(scalex)
    # decision function # confidence
    decision_value = model_knn.predict_proba(scalex).flatten()
    labels = model_knn.classes_
    # probability
    z = scipy.stats.zscore(decision_value)
    prob_value = scipy.special.softmax(z)
    
    # top 4
    top_4_prob_ind = prob_value.argsort()[::-1][:4]
    top_labels = labels[top_4_prob_ind]
    top_prob = prob_value[top_4_prob_ind]
    # put in dictornary
    top_dict = dict()
    for key,val in zip(top_labels,top_prob):
        top_dict.update({key:np.round(val,3)})
    
    return top_dict

@app.route('/tentang/')
def tentang():
    return render_template('tentang.html', judul='Tentang')

@app.route('/petunjuk/')
def petunjuk():
    return render_template('petunjuk.html', judul='Petunjuk')

if __name__ == "__main__":
    app.run(debug=True)