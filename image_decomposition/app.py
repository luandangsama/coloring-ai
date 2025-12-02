from flask import Flask, request, send_file, render_template
from io import BytesIO
from PIL import Image
from process_image import process_image, image_decomposition  # import your function
from datetime import datetime
import os
app = Flask(__name__)

@app.route("/")
def index():
    # Render the HTML page
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    td = datetime.now()
    # Get uploaded file
    file = request.files.get("image")
    if file is None:
        return "No file uploaded", 400

    img = Image.open(file.stream).convert("RGB")
    ## save to a temporary location
    temp_path = f"data/{td.strftime('%Y%m%d%H%M%S')}.png"
    img.save(temp_path)

    out_img = image_decomposition(temp_path)
    out_img.save(f"data/{td.strftime('%Y%m%d%H%M%S')}_processed.png")

    # Save output to in-memory buffer
    buf = BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)

    # Send image back to browser
    return send_file(buf, mimetype="image/png")

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    app.run(debug=False, port=5001)
