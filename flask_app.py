from flask import Flask,render_template
from flask import request

import numpy as np
import scipy
import os
import skimage
import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import pickle

app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')
MODEL_PATH = os.path.join(BASE_PATH,'static/models/')

model_sgd_path = os.path.join(MODEL_PATH,'Image_Classification_SGD.pkl')
scalar_path = os.path.join(MODEL_PATH,'Scalar.pkl')

model_sgd = pickle.load(open(model_sgd_path,'rb'))
scalar = pickle.load(open(scalar_path,'rb'))

#lets integrate everything into a single function
def pipe_model(path,scalar,model_sgd):
    #1) Read Image
    image = skimage.io.imread(path)
    #2) Rescale Image into 80x80
    resize_image = skimage.transform.resize(image,(80,80))
    rescale_image = 255*resize_image
    transform_image = rescale_image.astype(np.uint8)
    #3)Both Classes - RGB2Gray and HOG
    gray = skimage.color.rgb2gray(transform_image)
    feature_vector = skimage.feature.hog(gray,orientations=11,pixels_per_cell=(10,10),cells_per_block=(2,2))
    #now transform feature vector
    scaled = scalar.transform(feature_vector.reshape(1,-1))
    #4)Pass to model
    result = model_sgd.predict(scaled)
    #get confidence intervals of all classes
    dec_val = model_sgd.decision_function(scaled).flatten()
    labels = model_sgd.classes_
    #cal z score
    probs = scipy.special.softmax(scipy.stats.zscore(dec_val))
    #lets get top 5 probabilities
    top_5_ind = probs.argsort()[::-1][:5]
    top_5_labs = labels[top_5_ind]
    top_5_probs = probs[top_5_ind]
    best_dict = dict()
    for key,val in zip(top_5_labs,top_5_probs):
        best_dict.update({key:np.round(val,3)})
    return best_dict

@app.errorhandler(404)
def error404(error):
    err_response = "[Error 404 Occurred] PAGE NOT FOUND"
    return render_template('error.html',err_response=err_response)

@app.errorhandler(405)
def error405(error):
    err_response = "[Error 405 Occurred] METHOD NOT ALLOWED"
    return render_template('error.html',err_response=err_response)

@app.errorhandler(500)
def error500(error):
    err_response = "[Error 500 Occurred] PAGE NOT FOUND"
    return render_template('error.html',err_response=err_response)

@app.route('/')
def home():
    img = skimage.io.imread('https://raw.githubusercontent.com/dhruvj014/Animal-Identifier-WebApp/main/pics/uploader_img4.png')
    h,w,_ = img.shape
    aspecrat = h/w
    new_height = aspecrat*1100
    return render_template('home.html',height=new_height)

@app.route('/about-project')
def about_project():
    return render_template('about-project.html')

@app.route('/about-me')
def about_me():
    return render_template('about-me.html')

@app.route('/identifier',methods=['GET','POST'])
def index():
    if request.method == 'POST':
        upload_file = request.files['image_name']
        fname = upload_file.filename
        print('The file name that is uploaded is : '+fname)
        #check for jpg png jpeg
        ext = fname.split('.')[-1]
        print('File is of type .'+ext)
        if ext.lower() in ['jpg','png','jpeg']:
            path_save = os.path.join(UPLOAD_PATH,fname)
            upload_file.save(path_save)
            print('File saved successfully. ')
            results = pipe_model(path_save,scalar,model_sgd)
            print(results)
            new_ht = getratio(path_save)
            return render_template('upload.html',fileupload=True,extension=False,data = results,imgname = fname, height = new_ht)

        else:
            print('Please Upload Files of type .jpg , .png or .jpeg')

            return render_template('upload.html',extension=True,fileupload=False)
    else:
        return render_template('upload.html',extension=False,fileupload=False)

def getratio(path):
    img = skimage.io.imread(path)
    h,w,_ = img.shape
    aspecrat = h/w
    new_height = aspecrat*300
    return new_height


if __name__ == "__main__":
    app.run(debug=True)