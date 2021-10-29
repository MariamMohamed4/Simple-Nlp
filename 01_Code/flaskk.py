from flask import Flask, render_template, request
import pickle



app = Flask(__name__)
#api = Api(app)

model = pickle.load(open("C:/Users/mariam/Desktop/ITI_Materials/model.pkl","rb"))
vectorizer = pickle.load(open("C:/Users/mariam/Desktop/ITI_Materials/Vectorizer.pkl","rb"))


@app.route('/<string:job>',methods=['GET'])

#   takes job as parameters, transforms into features
#   predicts based on features and then return prediction

def returnJob(job):
    X = vectorizer.transform([job])
    pred = model.predict(X)
    #y = le.inverse_transform(model.predict(X))
    return {'Industry: ': pred[0]}

if __name__ == '__main__':
    app.run(debug=True)

