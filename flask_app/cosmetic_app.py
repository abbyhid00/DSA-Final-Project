import pickle
import numpy as np
# we are going to use the Flask micro web framework
from flask import Flask, request, jsonify

app = Flask(__name__)

def load_model():
    # unpickle header,posteriors and priors
    infile = open("flask_app/naive.p", "rb")
    header, posteriors, priors= pickle.load(infile)
    infile.close()
    return header, posteriors, priors
        
def naive_predict(header,posteriors, priors,instance):
    diff_probabilities = {}
    for drug in priors: #iterating through each drug
        diff_probabilities[drug] = np.log(priors[drug])
        for attribute, value in zip(header, instance):
            if attribute in posteriors[drug]:
                diff_probabilities[drug] += np.log(posteriors[drug][attribute][value])
            else: 
                diff_probabilities[drug] += np.log(1e-6)
    return max(diff_probabilities, key=diff_probabilities.get())

# we need to add some routes!
# a "route" is a function that handles a request
# e.g. for the HTML content for a home page
# or for the JSON response for a /predict API endpoint, etc
@app.route("/")
def index():
    # return content and status code
    return "<h1>Welcome to the interview predictor app</h1>", 200
        

# lets add a route for the /predict endpoint
@app.route("/predict")
def predict():
    # lets parse the unseen instance values from the query string
    # they are in the request object
    # product = request.args.get("ProductName") # defaults to None
    company = request.args.get("CompanyName")
    brand = request.args.get("BrandName")
    primary = request.args.get("PrimaryCategory")
    sub = request.args.get("SubCategory")
    instance = [company, brand, primary, sub]
    print(instance)
    header, posteriors, priors = load_model()
    # lets make a prediction!
    pred = naive_predict(header,posteriors,priors,instance)
    print(pred)
    if pred is not None:
        return jsonify({"prediction": pred}), 200
    # something went wrong!!
    return "Error making a prediction", 400


if __name__ == "__main__":
    header, posteriors,priors= load_model()
    app.run(host="0.0.0.0", port=5001, debug=False)