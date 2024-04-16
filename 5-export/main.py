from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load unique commune names to improve performance
# Ensure consistent string formatting by stripping leading/trailing whitespaces and converting to a common case (e.g., all lowercase)
commune_names = pd.read_csv('predictions.csv', usecols=['Libellé de la commune'])
commune_names['Libellé de la commune'] = commune_names['Libellé de la commune'].str.strip().str.lower()
commune_names = commune_names.drop_duplicates().sort_values('Libellé de la commune')

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_commune = request.form.get('commune_name', '').strip().lower()  # Normalize input to match data cleaning
    predictions_html = ""

    if selected_commune:
        print(f'Loading predictions for {selected_commune}...')
        # Load the full dataset
        predictions = pd.read_csv('predictions.csv')
        predictions['Libellé de la commune'] = predictions['Libellé de la commune'].str.strip().str.lower()
        
        # Filter data
        filtered_predictions = predictions[predictions['Libellé de la commune'] == selected_commune]
        if not filtered_predictions.empty:
            predictions_html = filtered_predictions.to_html(index=False)
        else:
            predictions_html = "<p>No data available for the selected commune.</p>"
            print(f"No data found for {selected_commune}")

    return render_template('index.html', commune_names=commune_names['Libellé de la commune'].tolist(), predictions=predictions_html, selected_commune=selected_commune)

if __name__ == '__main__':
    app.run(debug=True)
