import pandas as pd
import tensorflow as tf
from flask import Flask, request, render_template
import pickle



app = Flask(__name__)
model = pickle.load(open('finalized_model.pkl', 'rb'))
model2 = pickle.load(open('svm_autism_model.pkl', 'rb'))
model3 = pickle.load(open('knn_autism_model.pkl','rb'))
model4 = pickle.load(open('naive_bayes_autism_model.pkl','rb'))
model5 = pickle.load(open('stochastic_autism_model.pkl','rb'))
graph=tf.get_default_graph()
                
@app.route('/')
def home():
    return render_template('index.html',data=[{"name":"NN"},{"name":"SVM"},{"name":"KNN"},{"name":"Naive Bayes"},{"name":"Stochastic"}])

@app.route('/predict',methods=['GET','POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    select = request.form.get("model")
    global graph
    with graph.as_default():
        true=0;
        false=0;
        data = pd.read_csv(request.files["data"]);
        X = pd.DataFrame(data.iloc[:, 0:4])
        if select=="NN" :
            prediction = model.predict(X)
            prediction=(prediction>0.5)
            for i in prediction:
                if(i):
                    true=true+1;
                else:
                    false=false+1;
            prob = true/(false+true) 
            prob= round(prob,2)
            print(prob)
            if (true>false):
                output="SENSORY DISORDER MIGHT BE PRESENT(CNN)"
            else:
                output="Sensory Disorder not present"
        elif select =="SVM":
            prediction = model2.predict(X)
            prediction=(prediction>0.5)
            for i in prediction:
                if(i):
                    true=true+1;
                else:
                    false=false+1;     
            prob = true/(false+true) 
            prob= round(prob,2)
            if (true>false):
                output="Sensory Disorder might be present(SVM)\n The predicted probability is " + str(prob) +. "\nThe model uses Neural Network to generate this report. \nThe accuracy of this model is 94%. The precision for a child having sensory disorder is 0.97. It can model complex, multivariate sensory time series data. According to the session data provided to the system the autistic patient might have a sensory disorder (Auditory and Visual) present."

            else:
                output="Sensory Disorder not present(SVM). The predicted probability is " + str(prob) +". \nThe model uses SVM to generate this report. \nThe accuracy of this model is 83%. The precision for a child not having sensory disorder is 0.79. According to the session data provided to the system the autistic patient don't. have a sensory disorder(Auditory and Visual) present."
        elif select == "KNN":
            prediction = model3.predict(X)
            prediction=(prediction>0.5)
            for i in prediction:
                if(i):
                    true=true+1;
                else:
                    false=false+1;
                
            if (true>false):
                output="Sensory Disorder might be present(KNN)"
            else:
                output="Sensory Disorder not present(KNN)"
        elif select=="Naive Bayes" :
            prediction = model4.predict(X)
            prediction=(prediction>0.5)
            for i in prediction:
                if(i):
                    true=true+1;
                else:
                    false=false+1;
                
            if (true>false):
               output="Sensory Disorder might be present(NB)"
            else:
                output="Sensory Disorder not present(NB)"
        else:
            prediction = model5.predict(X)
            prediction=(prediction>0.5)
            for i in prediction:
                if(i):
                    true=true+1;
                else:
                    false=false+1;
                
            if (true>false):
                output="Sensory Disorder might be present(Stochastic)"
            else:
                output="Sensory Disorder not present(Stochastic)"

    return render_template('index.html', prediction_text='{}'.format(output))


    

if __name__ == "__main__":
    app.run(debug=True)
