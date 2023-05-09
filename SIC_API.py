
from flask import Flask,jsonify,request,render_template
from SIC_model import TextClassification

app = Flask(__name__)
tc = TextClassification()


###########################
@app.route("/")
def home():
    return render_template('predict.html')



# @app.route("/get_data")
# def getdata():
#     tc.model_getdata()
#     data = {'page':'get data page','message':'ok'}
#     return jsonify(data)



# @app.route("/preprocessing")
# def processing():
#     tc.model_processing()
#     data = {'page':'processing page','message':'ok'}
#     return jsonify(data)



# @app.route("/train")
# def train():
#     tc.model_train()
#     return render_template('result.html' ,message='model trained successfully')



# @app.route("/evaluate")
# def evaluate():
#     loss, acc = tc.model_evaluate()
#     return render_template('result.html',message = 'Your model accuracy is '+str(acc)+'And loss is '+str(loss))



# @app.route("/predict_gui")
# def predict_gui():   
#     return render_template('predict.html')


@app.route("/predict")
def predict():
    
    text = request.args['text']
    result = tc.model_predict(text)
    return render_template('predict.html',message = 'Your text type is: '+str(result))
###########################
app.run()

