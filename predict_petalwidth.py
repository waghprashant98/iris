import numpy as np
import pandas as pd
import pickle
from flask import Flask,jsonify,request
app=Flask(__name__)

lm=pickle.load(open("linear_model_regession.pkl","rb"))

@app.route("/predictpetalwidth")

def predictpetalwidth():
    data=request.get_json()
    SepalLengthCm=data["SepalLengthCm"]
    PetalLengthCm=data["PetalLengthCm"]
    Species=data["Species"] 
    
    def converter(Species):
        if Species=="setosa ":
            return 0
        elif Species=="versicolor":
            return 1
        else:
            return 2
    Species=converter(Species)

    test_df=pd.DataFrame({'SepalLengthCm':[SepalLengthCm],'PetalLengthCm':[PetalLengthCm],'Species':[Species]})
    pred_test_df=lm.predict(test_df)
    petal_width=np.around(pred_test_df[0],2)
    return jsonify("Predicted petalwidth :",petal_width)

if __name__=="__main__":
    app.run(debug=True)

    # nice
    #work