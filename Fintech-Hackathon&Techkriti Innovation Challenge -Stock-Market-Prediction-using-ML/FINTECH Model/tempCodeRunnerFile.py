from flask import Flask, render_template
import backup_training_loss_1

app = Flask(__name__)

@app.route('/')
def home():
    name = backup_training_loss_1.cp()
    return render_template('index.html', price = name)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)
