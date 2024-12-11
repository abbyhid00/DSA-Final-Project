import requests # a lib for making http requests
import json # a lib for working with json

url = "https://flask-app-demo.onrender.com/predict?ProductName=Adorned%20in%20Diamonds&CompanyName=Nail%20Alliance%2CLLC&BrandName=Morgan%20Taylor&PrimaryCategory=Nail%20Products&SubCategory=Nail%20Polish%20and%20Enamel"

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