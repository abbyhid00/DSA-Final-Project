import requests # a lib for making http requests
import json # a lib for working with json

url = "http://127.0.0.1:5001/predict?CompanyName=Bliss+World+LLC&BrandName=Bliss&PrimaryCategory=Skin+Care+Products&SubCategory=Skin+Cleansers"

response = requests.get(url=url)

# first thing, check the response's status_code
# https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods
# https://developer.mozilla.org/en-US/docs/Web/HTTP/Status#successful_responses
print(response.status_code)
if response.status_code == 200:
    # STATUS OK
    # we can extract the prediction from the response's JSON text
    json_object = json.loads(response.text)
    print(json_object)
    pred = json_object["prediction"]
    print("prediction:", pred)
else: 
    print("Error message")