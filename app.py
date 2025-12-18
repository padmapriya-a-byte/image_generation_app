from flask import Flask, render_template, request
from diffusers import StableDiffusionPipeline

app = Flask(__name__)

# Load model once (CPU-safe)
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)

@app.route("/", methods=["GET", "POST"])
def index():
    image_path = None

    if request.method == "POST":
        prompt = request.form.get("prompt")

        if prompt:
            image = pipe(
                prompt,
                num_inference_steps=20   # reduced for CPU
            ).images[0]

            image_path = "static/generated_image.png"
            image.save(image_path)

    return render_template("index.html", image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
