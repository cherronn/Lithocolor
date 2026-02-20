import io
import base64

import matplotlib
matplotlib.use("Agg")  # required for servers (no display)

import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file
from PIL import Image

from lithocolor_core import image_to_heightmap

app = Flask(__name__)

def _img_bytes_to_b64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("utf-8")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    if "image" not in request.files:
        return "No file uploaded", 400

    f = request.files["image"]
    img = Image.open(f.stream)

    hm = image_to_heightmap(img)
    hm_img = Image.fromarray(hm, mode="L")

    # original -> PNG bytes
    orig_buf = io.BytesIO()
    img.convert("RGB").save(orig_buf, format="PNG")
    orig_bytes = orig_buf.getvalue()

    # heightmap -> PNG bytes
    hm_buf = io.BytesIO()
    hm_img.save(hm_buf, format="PNG")
    hm_bytes = hm_buf.getvalue()

    # histogram -> PNG bytes
    fig, ax = plt.subplots()
    ax.hist(hm.ravel(), bins=256)
    ax.set_title("Linearized Height Map")
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    ax.grid(True)

    hist_buf = io.BytesIO()
    fig.savefig(hist_buf, format="png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    hist_bytes = hist_buf.getvalue()

    return render_template(
        "index.html",
        original_b64=_img_bytes_to_b64(orig_bytes),
        heightmap_b64=_img_bytes_to_b64(hm_bytes),
        hist_b64=_img_bytes_to_b64(hist_bytes),
    )

@app.route("/download/heightmap", methods=["POST"])
def download_heightmap():
    if "image" not in request.files:
        return "No file uploaded", 400

    img = Image.open(request.files["image"].stream)
    hm = image_to_heightmap(img)
    hm_img = Image.fromarray(hm, mode="L")

    buf = io.BytesIO()
    hm_img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(
        buf,
        mimetype="image/png",
        as_attachment=True,
        download_name="lithocolor_heightmap.png",
    )

@app.route("/download/histogram", methods=["POST"])
def download_histogram():
    if "image" not in request.files:
        return "No file uploaded", 400

    img = Image.open(request.files["image"].stream)
    hm = image_to_heightmap(img)

    fig, ax = plt.subplots()
    ax.hist(hm.ravel(), bins=256)
    ax.set_title("Linearized Height Map")
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    ax.grid(True)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    buf.seek(0)
    return send_file(
        buf,
        mimetype="image/png",
        as_attachment=True,
        download_name="lithocolor_histogram.png",
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
