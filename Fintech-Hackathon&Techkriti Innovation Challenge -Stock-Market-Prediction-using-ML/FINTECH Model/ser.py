from flask import Flask, render_template
import backup_training_loss_1 as bt

app = Flask(__name__)

@app.route('/')
def home():
    name1, name, cName, d1 = bt.pp()
    return render_template('index.html',accuracy = name1,
                           price = name, company = cName ,date = d1)

if __name__ == '__main__':
    app.run(port = 5000)
