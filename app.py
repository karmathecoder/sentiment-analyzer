from flask import Flask,render_template,request
from class_function import link_to_result, wordcloud_gen


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/results',methods=['GET'])
def result():    
    url = request.args.get('url')
    productname = request.args.get('num')
    res_comp = link_to_result(url,productname)
    # res_comp = pd.read_csv('process/result.csv')

    d = []
    for index, row in res_comp.iterrows():
        x = {}
        x['review'] = row["answer_option"]
        x['score'] = round(row["review_score"]*100,2)
        d.append(x)
    
    wordcloud_gen(res_comp)
    return render_template('result.html',dic=d,proname=productname)
    
    
@app.route('/wc')
def wc():
    return render_template('wc.html')



if __name__ == '__main__':
    app.run(debug=True)