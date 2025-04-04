from flask import Flask, render_template, request
from stock_predictor import train_and_predict

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print("Form submitted!")  # Debug: Check if POST is received
        try:
            # Get form data
            company = request.form['company'].upper()
            start_date = request.form['start_date']
            end_date = request.form['end_date']
            test_start = request.form['test_start']
            test_end = request.form['test_end']
            steps_ahead = int(request.form['steps_ahead'])
            print(f"Inputs: {company}, {start_date}, {end_date}, {test_start}, {test_end}, {steps_ahead}")  # Debug

            # Validate dates
            current_date = '2025-04-03'
            if test_end > current_date:
                test_end = current_date
                print(f"Adjusted test_end to {test_end}")  # Debug

            # Run Experiment 1: ARIMA + LSTM
            print("Running Experiment 1: ARIMA + LSTM")  # Debug
            mse_results_arima_lstm, plot_images_arima_lstm = train_and_predict(
                company=company,
                start_date=start_date,
                end_date=end_date,
                test_start=test_start,
                test_end=test_end,
                steps_ahead=steps_ahead,
                layer_type='LSTM',
                use_sarima=False
            )
            print("Experiment 1 completed")  # Debug

            # Run Experiment 2: SARIMA + GRU
            print("Running Experiment 2: SARIMA + GRU")  # Debug
            mse_results_sarima_gru, plot_images_sarima_gru = train_and_predict(
                company=company,
                start_date=start_date,
                end_date=end_date,
                test_start=test_start,
                test_end=test_end,
                steps_ahead=steps_ahead,
                layer_type='GRU',
                use_sarima=True
            )
            print("Experiment 2 completed")  # Debug

            print("Rendering results page")  # Debug
            result = render_template('results.html',
                                    company=company,
                                    mse_results_arima_lstm=mse_results_arima_lstm,
                                    plot_images_arima_lstm=plot_images_arima_lstm,
                                    mse_results_sarima_gru=mse_results_sarima_gru,
                                    plot_images_sarima_gru=plot_images_sarima_gru,
                                    steps_ahead=steps_ahead)
            print("Template rendered successfully")  # Debug
            return result

        except Exception as e:
            print(f"Error occurred: {str(e)}")  # Debug
            return render_template('index.html', error=str(e))
    
    print("Rendering index page")  # Debug
    return render_template('index.html', error=None)

if __name__ == '__main__':
    app.run(debug=True)