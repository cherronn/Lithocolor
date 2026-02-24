import io
import base64
import os
from werkzeug.utils import secure_filename

import matplotlib
matplotlib.use("Agg")  # required for servers (no display)

import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file
from PIL import Image

from lithocolor_core import image_to_heightmap

# Render Free friendly limits
MAX_MEGA_PIXELS = 25  # 25 MP is safe on 512 MB
MAX_PIXELS = MAX_MEGA_PIXELS * 1_000_000
MAX_EDGE = 6000       # cap longest side

# Allow loading, but we will downscale ourselves
Image.MAX_IMAGE_PIXELS = 100_000_000


def _load_and_shrink(upload) -> Image.Image:
    """
    Load an uploaded image safely and downscale it to avoid memory/timeouts.
    Returns an RGB PIL Image.
    """
    img = Image.open(upload.stream)
    img = img.convert("RGB")

    w, h = img.size
    pixels = w * h

    # If either dimension is too large, downscale by edge first
    if max(w, h) > MAX_EDGE:
        img.thumbnail((MAX_EDGE, MAX_EDGE), Image.Resampling.LANCZOS)
        w, h = img.size
        pixels = w * h

    # If still too many pixels, scale down by area
    if pixels > MAX_PIXELS:
        scale = (MAX_PIXELS / float(pixels)) ** 0.5
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return img


app = Flask(__name__)


def _out_name(uploaded_filename: str, suffix: str, ext: str) -> str:
    """
    Turn "my photo.jpg" into "my_photo_lithocolor.png" (safe for browsers).
    """
    base = secure_filename(uploaded_filename or "image")
    stem, _ = os.path.splitext(base)
    if not stem:
        stem = "image"
    return f"{stem}_{suffix}{ext}"


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
    img = _load_and_shrink(f)

    hm = image_to_heightmap(img)
    hm_img = Image.fromarray(hm, mode="L")

    # original -> PNG bytes
    orig_buf = io.BytesIO()
    img.save(orig_buf, format="PNG")
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

    f = request.files["image"]
    img = _load_and_shrink(f)

    hm = image_to_heightmap(img)
    hm_img = Image.fromarray(hm, mode="L")

    buf = io.BytesIO()
    hm_img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(
        buf,
        mimetype="image/png",
        as_attachment=True,
        download_name=_out_name(f.filename, "lithocolor", ".png"),
    )


@app.route("/download/histogram", methods=["POST"])
def download_histogram():
    if "image" not in request.files:
        return "No file uploaded", 400

    f = request.files["image"]
    img = _load_and_shrink(f)

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
        download_name=_out_name(f.filename, "histogram", ".png"),
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
