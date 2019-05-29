#thanks:
# https://medium.freecodecamp.org/how-to-build-a-web-application-using-flask-and-deploy-it-to-the-cloud-3551c985e492
#https://towardsdatascience.com/develop-a-nlp-model-in-python-deploy-it-with-flask-step-by-step-744f3bdd7776
#https://medium.com/datadriveninvestor/deploy-your-pytorch-model-to-production-f69460192217?FGa=true
# http://flask.pocoo.org/docs/1.0/patterns/fileuploads/#uploading-files

import os
from pathlib import Path

"""importing AI App"""
from check_flower import check_flower

"""importing the Flask module 
and creating a Flask web server from the Flask module"""
from flask import Flask, render_template, url_for, request, flash, redirect, send_from_directory

"""there is that principle called “never trust user input”. 
This is also true for the filename of an uploaded file. 
All submitted form data can be forged, and filenames can be dangerous."""
from werkzeug.utils import secure_filename

"""The UPLOAD_FOLDER is where we will store the uploaded files"""
UPLOAD_FOLDER = Path('C:/Users/Justyna/GIT/flask/uploads/')

"""The ALLOWED_EXTENSIONS is the set of allowed file extensions"""
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

""" creating an instance of the Flask class and calling it app. 
Here we are creating a new web application. __name__  means 
this current file. In this case, it is main.py. 
This current file will represent a web application"""
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

"""It represents the default page. For example, 
if we go to a website such as “google.com/” with nothing after 
the slash. Then this will be the default page of Google."""
@app.route("/")
#If the user goes to my website and they go to the default page (nothing after the slash), then this function will get activated:
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/result",methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        #checks if the post request has the file part
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        image = request.files['image'] #.get('image')
        #if user does not select file, browser also
        #submit an empty part without filename
        if image.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if image and allowed_file(image.filename):
            imagename = secure_filename(image.filename) #always we need to use that 
                                             #function to secure a filename before storing it 
                                             # directly on the filesystem.
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], imagename)) #uses
                #the save() method of the file to save the file permanently on the
                #file system
            
            # image_path = redirect(url_for('uploaded_file', filename=imagename))
            fname, fprob, fclass = check_flower(image)
            fname = fname.upper()
            fprob = '{:05.2f}'.format(fprob * 100)
            fclass = str(fclass)
            ipath = 'images/'+fclass+'/image.jpg'
            # ipath = 'images/11/image.jpg'
            
            return render_template('result.html', image_name=imagename, 
                                        p_name=fname, prob=fprob, ipath=ipath)
    return

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Returns the file of the name <filename>
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


"""When we run main.py, it will change its name to __main__
and only then will that if statement activate."""
if __name__ == "__main__":
    app.run(debug=True) #runs the application
"""Having debug=True allows possible Python errors to appear
on the web page. This will help us trace the errors."""

