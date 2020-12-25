from flask import *
from svc_model import pred_clean

app = Flask(__name__, static_url_path='', static_folder='./templates', template_folder='./templates')

@app.route('/', methods=['GET', 'POST'])
def basic():
    #request.method and request.form
    if request.method == 'POST':
        string = request.form['name']
        classe, proba0, proba1 = pred_clean(string)

        return render_template('index.html', variable=classe, variable0=proba0, variable1=proba1)

    return render_template('index.html')

if __name__ == '__main__':
    app.run()
