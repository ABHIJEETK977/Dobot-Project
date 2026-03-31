import base64
from openai import OpenAI

client = OpenAI()

prompt = """
Convert the portrait into sparse robotic-trace line art. Hard constraints: - White background. - Thin black outlines only.
- No filled black areas.
- No solid hair fill.
- No solid eyebrow fill.
- No solid beard fill.
- Hair should be shown only with: 1. outer boundary of the hairstyle 2. 3 to 8 major flow lines
- Leave the interior of the hair mostly white.
- Facial hair should be reduced to a few contour strokes only.
- Keep the drawing minimal and structurally readable.
- The result must look like pen outline drawing, not a stencil or silhouette cutout.
"""

with open("input.jpg", "rb") as img:
    result = client.images.edit(
        model="gpt-image-1.5",
        image=img,
        prompt=prompt,
        input_fidelity="high",   # helps preserve facial/style features
        size="1024x1024",
        output_format="png",
    )

image_b64 = result.data[0].b64_json
image_bytes = base64.b64decode(image_b64)

with open("output.png", "wb") as f:
    f.write(image_bytes)

print("Saved output.png")
