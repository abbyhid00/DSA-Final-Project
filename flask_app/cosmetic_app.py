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
    predictions = []
    unique_labels = list(priors.keys())  #get labels from priors

    class_probabilities = {}
    for label in unique_labels:  #calculate the probability of each class given the input x
        prob = priors[label]
        class_index = unique_labels.index(label)
        for feature_idx, feature_val in enumerate(instance):
            feature_key = f"att{feature_idx + 1}" #feature value exists in the training data, use its probability
            if feature_key in posteriors and feature_val in posteriors[feature_key]:
                prob *= posteriors[feature_key][feature_val][class_index]
        class_probabilities[label] = prob
    predicted_label = max(class_probabilities, key=class_probabilities.get) #predict using label with highest probability
    predictions.append(predicted_label)

    return predictions


# we need to add some routes!
# a "route" is a function that handles a request
# e.g. for the HTML content for a home page
# or for the JSON response for a /predict API endpoint, etc
@app.route("/")
def index():
    # return content and status code
    return "<h1>Welcome to the cosmetic chemical predictor app</h1>", 200
        

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