from flask import Flask

if __name__ == '__main__':
    app = Flask(__name__)
    app.run(ssl_context='adhoc', host='0.0.0.0', port=443)
