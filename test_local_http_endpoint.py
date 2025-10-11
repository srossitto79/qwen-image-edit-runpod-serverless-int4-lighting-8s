import requests
import base64

url = "http://localhost:3000/"

base64_input = ""
with open("./test_input.jpg", "rb") as f:
    img_bytes = f.read()
    base64_input = base64.b64encode(img_bytes).decode("utf-8")
    #base64_input = f"image/jpeg;base64,{base64_input}"

payload = {
    "input": {
        "image": base64_input,
        "prompt": "Transform this exterior space, enhancing textures and lighting. Preserve the camera angle, geometry, and composition. Enhance the vibrant colors and detailed textures of the image. Add subtle shadows and highlights to create a more realistic and immersive environment. Preserve the natural atmosphere and the ambiance.",
        "num_inference_steps": 20,
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
