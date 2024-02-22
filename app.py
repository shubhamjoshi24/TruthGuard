from flask import Flask, escape, request, render_template
import pickle

vector = pickle.load(open("vectorizer.pkl", 'rb')) 
model = pickle.load(open("finalized_model.pkl", 'rb'))
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("prediction.html")

@app.route("/prediction",methods=['GET','POST'])
def prediction():
    if request.method == "POST":
        news = str(request.form['news'])
        
        predict = model.predict(vector.transform([news]))
        print(predict)
       
        if(predict=='REAL'):
            output="News is real"
        elif(predict=='FAKE'):
            output="News is fake"
        return render_template("prediction.html", result=output)
    else:
        return render_template("prediction.html")
    




if __name__ == '__main__':
    app.run()