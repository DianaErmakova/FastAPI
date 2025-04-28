import requests

image_path = "img.png"

with open(image_path, "rb") as image_file:
    files = {"file": image_file}

    response = requests.post("http://127.0.0.1:8000/predict/", files=files)

print(response.json())
