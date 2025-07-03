from flask import Flask, render_template, request
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model (assuming it's a pickle file)
model = pickle.load(open('logistic_regression_model.pkl', 'rb'))

# Define the expected column names in the exact order as used during model training
expected_columns = [
    'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot',
    'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
]

@app.route('/')
def home():
    """Render the home page"""
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handle prediction logic and render result"""
    if request.method == 'POST':
        # Collect data from form
        age = int(request.form['age'])
        bp = int(request.form['bp'])
        sg = float(request.form['sg'])
        al = int(request.form['al'])
        su = int(request.form['su'])
        rbc = request.form.get('rbc', 'normal')
        pc = request.form.get('pc', 'normal')
        pcc = request.form.get('pcc', 'notpresent')
        ba = request.form.get('ba', 'notpresent')
        bgr = float(request.form.get('bgr', 0))
        bu = float(request.form.get('bu', 0))
        sc = float(request.form.get('sc', 0))
        sod = float(request.form.get('sod', 0))
        pot = float(request.form.get('pot', 0))
        hemo = float(request.form.get('hemo', 0))
        pcv = float(request.form.get('pcv', 0))
        wc = float(request.form.get('wc', 0))
        rc = float(request.form.get('rc', 0))
        htn = request.form.get('htn', 'no')
        dm = request.form.get('dm', 'no')
        cad = request.form.get('cad', 'no')
        appet = request.form.get('appet', 'good')
        pe = request.form.get('pe', 'no')
        ane = request.form.get('ane', 'no')

        # Mapping categorical to numerical
        rbc_map = {'normal': 0, 'abnormal': 1}
        pc_map = {'normal': 0, 'abnormal': 1}
        pcc_map = {'notpresent': 0, 'present': 1}
        ba_map = {'notpresent': 0, 'present': 1}
        htn_map = {'no': 0, 'yes': 1}
        dm_map = {'no': 0, 'yes': 1}
        cad_map = {'no': 0, 'yes': 1}
        appet_map = {'good': 0, 'poor': 1}
        pe_map = {'no': 0, 'yes': 1}
        ane_map = {'no': 0, 'yes': 1}

        # Create a DataFrame with the exact expected feature order
        input_data = pd.DataFrame([[
            age, bp, sg, al, su,
            rbc_map.get(rbc, 0),
            pc_map.get(pc, 0),
            pcc_map.get(pcc, 0),
            ba_map.get(ba, 0),
            bgr, bu, sc, sod, pot,
            hemo, pcv, wc, rc,
            htn_map.get(htn, 0),
            dm_map.get(dm, 0),
            cad_map.get(cad, 0),
            appet_map.get(appet, 0),
            pe_map.get(pe, 0),
            ane_map.get(ane, 0)
        ]], columns=expected_columns)

        # Make prediction
        prediction = model.predict(input_data)[0]
        result = 'CKD (Chronic Kidney Disease)' if prediction == 1 else 'Non-CKD'

        return render_template('result.html', prediction=result)

    # For GET request
    return render_template('indexnew.html')


if __name__ == '__main__':
    app.run(debug=True)
