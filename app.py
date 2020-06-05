import pandas as pd
import tensorflow as tf
from flask import Flask, request, render_template
import pickle
from bokeh.plotting import figure
from bokeh.embed import components

app = Flask(__name__)
model = pickle.load(open('finalized_model.pkl', 'rb'))
graph=tf.get_default_graph()

data1 = pd.read_csv('Autism_Data - HT3.csv', names=['stimuli','heart','temp', 'facial','target'])
target= data1['target']
feature_names = data1.columns[0:-1].values.tolist() 

def create_figure(current_feature_name, bins):
    p = figure(title=current_feature_name, x_axis_label = current_feature_name, y_axis_label = 'Count')
    p.line(current_feature_name, 'target', line_width=2)
    return p  
                
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    global graph
    with graph.as_default():
        true=0;
        false=0;
        #int_features = [x for x in request.form.values()]
        data = pd.read_csv(request.files["data"]);
        current_feature_name = request.args.get("feature_name")
        if current_feature_name == None:
            current_feature_name = "heart"
        plot = create_figure(current_feature_name, 10)
        script, div = components(plot)
        #tshow(p)
        #data.save(os.path.join("", data.filename))
        X = pd.DataFrame(data.iloc[:, 0:4])
        prediction = model.predict(X)
        prediction=(prediction>0.5)
        for i in prediction:
            if(i):
                true=true+1;
            else:
                false=false+1;
                
        if (true>false):
            output="Sensory Disorder might be present"
        else:
            output="Sensory Disorder not present"

    return render_template('index.html', prediction_text='{}'.format(output),script=script, div=div,
		feature_names=feature_names,  current_feature_name=current_feature_name)
    

if __name__ == "__main__":
    app.run(debug=True)