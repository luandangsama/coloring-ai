from flask import Flask, request, send_file, render_template
from io import BytesIO
from PIL import Image
from process_image import process_image, image_decomposition  # import your function
from datetime import datetime
import os
import zipfile
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

@app.route("/process-zip", methods=["POST"])
def process_zip():
    """
    Returns a ZIP file containing:
      - processed.png : the processed image
      - report.txt    : some information about the image
    """
    file = request.files.get("image")
    if file is None:
        return "No file uploaded", 400

    img = Image.open(file.stream).convert("RGB")

    # 🔧 Process image with your function
    out_img = process_image(img)

    # 1) Get processed image bytes (PNG)
    img_buf = BytesIO()
    out_img.save(img_buf, format="PNG")
    img_bytes = img_buf.getvalue()

    # 2) Build a text report (replace with your real info)
    info_lines = [
        "Processed image report",
        "----------------------",
        f"Original size: {img.width}x{img.height}",
        f"Processed size: {out_img.width}x{out_img.height}",
        f"Mode: {out_img.mode}",
    ]
    report_text = "\n".join(info_lines)
    report_bytes = report_text.encode("utf-8")

    # 3) Pack everything into a ZIP in memory
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("processed.png", img_bytes)
        zf.writestr("report.txt", report_bytes)

    zip_buf.seek(0)

    return send_file(
        zip_buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name="results.zip",
    )

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    app.run(debug=False, port=5001)
