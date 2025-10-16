import requests
import base64
from io import BytesIO
import os
import time

url = "https://api.runpod.ai/v2/s8irofmg23ebgx/run"

base64_input = ""
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
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

payload = {
    "input": {
        "image": base64_input,
        #"prompt": "Transform this exterior space, enhancing textures and lighting. Preserve the camera angle, geometry, and composition. Enhance the vibrant colors and detailed textures of the image. Add subtle shadows and highlights to create a more realistic and immersive environment. Preserve the natural atmosphere and the ambiance.",
        "prompt": "Transform current image into a photorealistic one. Upgrade this exterior design image. Maintain architectural massing while refining materials, landscaping, and lighting to feel photorealistic. Style focus: premium-photography.",
        "num_inference_steps": 8,
        "true_cfg_scale": 4.0
    }
}

# Submit job
headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
resp = requests.post(url, json=payload, headers=headers)
resp.raise_for_status()

job_data = resp.json()
job_id = job_data["id"]
print(f"✅ Job submitted: {job_id}")

# Poll for status
status_url = f"https://api.runpod.ai/v2/s8irofmg23ebgx/status/{job_id}"
max_wait = 600  # 10 minutes max
start_time = time.time()

while time.time() - start_time < max_wait:
    status_resp = requests.get(status_url, headers=headers)
    status_resp.raise_for_status()
    status_data = status_resp.json()
    
    status = status_data.get("status")
    print(f"Status: {status}")
    
    if status == "COMPLETED":
        result = status_data.get("output")
        if result and "image_base64" in result:
            img_b64 = result["image_base64"]
            img_bytes = base64.b64decode(img_b64)
            with open("result.png", "wb") as f:
                f.write(img_bytes)
            print("✅ Image saved to result.png")
            break
        else:
            raise ValueError(f"Unexpected output: {result}")
    elif status == "FAILED":
        raise ValueError(f"Job failed: {status_data}")
    
    time.sleep(2)  # Wait 2 seconds before polling again
else:
    raise TimeoutError(f"Job {job_id} did not complete within {max_wait} seconds")
