from flask import Flask, request, send_file, render_template
from io import BytesIO
from PIL import Image
from process_image import process_image, image_decomposition, image_to_svgs  # import your function
from datetime import datetime
import os
import zipfile
import numpy as np
import cv2
app = Flask(__name__)
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

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
    cv2_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    out_img = image_decomposition(cv2_image)

    # Save output to in-memory buffer
    buf = BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)

    # Send image back to browser
    return send_file(buf, mimetype="image/png")

@app.route("/process-zip", methods=["POST"])
def process_zip():
    td = datetime.now()
    save_dir  = td.strftime('%Y%m%d%H%M')
    os.makedirs(f"data/{save_dir}/svgs", exist_ok=True)

    file = request.files.get("image")
    if file is None:
        return "No file uploaded", 400
    
    img = Image.open(file.stream).convert("RGB")
    ## save to a temporary location
    temp_path = f"data/{save_dir}/original.png"
    img.save(temp_path)

    cv2_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    out_img = image_decomposition(cv2_image)
    out_path = f"data/{save_dir}/decomposed.png"
    out_img.save(out_path)

    output_dir = f"data/{save_dir}/svgs"
    image_to_svgs(temp_path, output_dir)

    SVG_DIR = BASE_DIR / "data" / save_dir / "svgs"
    svg_files = list(SVG_DIR.glob("*.svg"))
    if not svg_files:
        return "No SVG files found on server", 404

    # 3) Pack everything into a ZIP in memory
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for svg_path in svg_files:
            # arcname controls the name inside the ZIP (no full path)
            zf.write(svg_path, arcname=svg_path.name)
    zip_buf.seek(0)

    return send_file(
        zip_buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"results_{save_dir}.zip",
    )

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    app.run(debug=False, port=5001)
