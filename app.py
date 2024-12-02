from flask import Flask, render_template, request, send_file
import os
import cv2
from ultralytics import YOLO
import supervision as sv
import pyresearch

# Initialize Flask app
app = Flask(__name__)

# Load YOLO model
model = YOLO("newmodel.pt")

# Set upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image processing function
def process_image(input_image_path: str, output_image_path: str):
    # Read the image
    image = cv2.imread(input_image_path)
    if image is None:
        print("Error: Unable to read the image.")
        return

    # Resize the image
    resized = cv2.resize(image, (640, 640))

    # Perform detection
    detections = sv.Detections.from_ultralytics(model(resized)[0])

    # Annotate the image
    annotated = sv.BoundingBoxAnnotator().annotate(scene=resized, detections=detections)
    annotated = sv.LabelAnnotator().annotate(scene=annotated, detections=detections)

    # Save the annotated image
    cv2.imwrite(output_image_path, annotated)
    print(f"Processed and saved: {output_image_path}")

# Route to handle image upload
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return 'No file part'
        
        file = request.files['file']
        
        # If no file is selected
        if file.filename == '':
            return 'No selected file'
        
        # If file is allowed
        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # Define output image path
            output_filename = os.path.join(app.config['OUTPUT_FOLDER'], 'annotated_' + file.filename)

            # Process the image
            process_image(filename, output_filename)

            # Return the processed image
            return send_file(output_filename, mimetype='image/jpeg')
    
    return '''
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Upload an Image</title>
        <style>
            body {
                font-family: 'Arial', sans-serif;
                background-color: #f1f1f1;
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            header {
                background-color: #333;
                color: white;
                padding: 20px;
                text-align: center;
                font-size: 24px;
            }
            footer {
                background-color: #333;
                color: white;
                padding: 15px;
                text-align: center;
                position: fixed;
                bottom: 0;
                width: 100%;
            }
            footer a {
                color: #ffcc00;
                text-decoration: none;
                margin: 0 15px;
                font-weight: bold;
            }
            footer a:hover {
                text-decoration: underline;
            }
            .container {
                max-width: 800px;
                margin: 40px auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            h1 {
                color: #333;
                text-align: center;
                font-size: 28px;
                margin-bottom: 20px;
            }
            .upload-form {
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            input[type="file"] {
                margin-bottom: 20px;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: #f8f8f8;
                font-size: 16px;
            }
            input[type="submit"] {
                padding: 15px 30px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 18px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }
            input[type="submit"]:hover {
                background-color: #45a049;
            }
            footer p {
                margin: 5px 0;
            }
        </style>
    </head>
    <body>
        <header>
            AI-Powered Chest X-Ray Analysis: Normal or Abnormal? - PyResearch
        </header>

        <div class="container">
            <h1>Upload an Image Chest X-Ray AI Detection: Normal vs Abnormal</h1>
            <form class="upload-form" method="POST" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required>
                <input type="submit" value="Upload Image">
            </form>
        </div>

        <footer>
            <p>Contact us:</p>
            <p>Phone: +966539723031</p>
            <p>
                <a href="https://www.youtube.com/channel/UCyB_7yHs7y8u9rONDSXgkCg/join" target="_blank">Channel Membership</a> |
                <a href="https://www.facebook.com/Pyresearch" target="_blank">Facebook</a> |
                <a href="https://www.youtube.com/c/Pyresearch" target="_blank">YouTube</a> |
                <a href="https://medium.com/@Pyresearch" target="_blank">Medium</a> |
                <a href="https://www.instagram.com/pyresearch/" target="_blank">Instagram</a> |
                <a href="https://www.linkedin.com/company/Pyresearch" target="_blank">LinkedIn</a> |
                <a href="https://twitter.com/Noorkhokhar10" target="_blank">Twitter</a> |
                <a href="https://discord.com/invite/BHxGBn98" target="_blank">Discord</a> |
                <a href="https://github.com/Pyresearch" target="_blank">GitHub</a> |
                <a href="https://www.quora.com/profile/Pyresearch" target="_blank">Quora</a> |
                <a href="https://github.com/noorkhokhar99" target="_blank">GitHub (Personal)</a>
            </p>
        </footer>
    </body>
    </html>
    '''

if __name__ == "__main__":
    # Create upload and output folders if they don't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    app.run(debug=True)
