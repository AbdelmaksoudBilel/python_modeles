# import requests

# url = "http://127.0.0.1:8000/predict_tsa"

# data = {
#     "features": '[0,1,0,0,1,0,1,0,0,1,5,"m",0,0]'
# }

# files = {
#     "image": open("data/images/test/Autistic/Autistic.0.jpg", "rb")
# }

# response = requests.post(url, data=data, files=files)

# print("Status code:", response.status_code)
# print("Raw response:")
# print(response.text)

import requests

url = "http://127.0.0.1:8000/predict_rm"

data = {
    "features": [5.0,2.0,2.0,1.0,5.0,1.0,1.0,1.0,5.0,4,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0]  # adapter à ton nombre exact features
}

response = requests.post(url, json=data)

print(response.json())