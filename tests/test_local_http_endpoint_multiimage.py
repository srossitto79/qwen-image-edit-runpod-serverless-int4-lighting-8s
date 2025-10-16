import requests
import base64
from io import BytesIO

url = "http://localhost:3000/"

base64_input = ""
base64_input2 = ""

with open("./test_input.jpg", "rb") as f:
    img_bytes = f.read()
    #resize the image to 1MPixels
    from PIL import Image
    img = Image.open(f)
    img = img.resize((int((1_000_000 * img.width / img.height) ** 0.5), int((1_000_000 * img.height / img.width) ** 0.5)))
    #img.save("resized.png")
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_bytes = buffer.getvalue()
    base64_input = base64.b64encode(img_bytes).decode("utf-8")

with open("./test_object.png", "rb") as f:
    img_bytes = f.read()
    #resize the image to 1MPixels
    from PIL import Image
    img = Image.open(f)
    img = img.resize((int((1_000_000 * img.width / img.height) ** 0.5), int((1_000_000 * img.height / img.width) ** 0.5)))
    #img.save("resized.png")
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_bytes = buffer.getvalue()
    base64_input2 = base64.b64encode(img_bytes).decode("utf-8")

payload = {
    "input": {
        "images": [base64_input, base64_input2],
        #"prompt": "Transform this exterior space, enhancing textures and lighting. Preserve the camera angle, geometry, and composition. Enhance the vibrant colors and detailed textures of the image. Add subtle shadows and highlights to create a more realistic and immersive environment. Preserve the natural atmosphere and the ambiance.",
        "prompt": "insert the bike from image2 into image1 in a photorealistic way. Transform current image into a photorealistic one. Upgrade this exterior design image. Maintain architectural massing while refining materials, landscaping, and lighting to feel photorealistic. Style focus: premium-photography.",
        "num_inference_steps": 8,
        "true_cfg_scale": 4.0
    }
}

# Send the request
resp = requests.post(url, json=payload)
resp.raise_for_status()

data = resp.json()

if "image_base64" not in data:
    raise ValueError(f"Unexpected response: {data}")

# Decode and save the resulting image
img_b64 = data["image_base64"]
img_bytes = base64.b64decode(img_b64)

with open("result.png", "wb") as f:
    f.write(img_bytes)

print("✅ Image saved to result.png")
