import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from torchvision.models import resnet50
from torchvision import transforms
import torch
from PIL import Image

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 's3cr37 53Y'

model = resnet50()
model.fc = torch.nn.Linear(2048,1)
model.load_state_dict(torch.load('resnet50v1.pth', weights_only=True))


# checks that uploaded file is among allowed file types
def allowed_file(filename):
    # checks for a '.' and one of the three image file types
    return '.' in filename and \
        filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

# evaluate
def predict_image(img, model):

    model.eval()

    # preprocess
    preprocess = transforms.Compose([transforms.Resize((224,224)), 
                                     transforms.ToTensor()])
    
    img = preprocess(img)
    
    # classify
    batch = img.unsqueeze(0) # convert shape [3,224,224] to [1,3,224,224]

    with torch.no_grad(): #don't track gradients during eval
        output = model(batch)
        probability = torch.sigmoid(output).item()

    if probability > 0.5:
        label = 1 # malignant
    else:
        label = 0 # benign
        probability = 1 - probability # ex: 0.0001 -> 0.9999

    probability *= 100
    probability = round(probability, 2)   

    return label, probability

@app.route('/', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url) # return to main page
        file = request.files['file'] # store file object in variable
        if file.filename == '': # checks for if the user did not select a file
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # ex: file 'bird.png' saves to uploads/bird.png on computer
            file.save(file_path)

            # load image from path as tensor
            img = Image.open(file_path).convert('RGB')

            # evaluate image
            index, score = predict_image(img, model)

            # delete image
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f'error deleting image from disk {e}')
            else:
                print(f'Path {file_path} not found')

            # display results
            return render_template('results.html', index=index, score=score)
            
        
    return render_template('index.html')

        

if __name__ == "__main__":
    app.run(port=8000, debug=True)