from data_fetching import fetch_stock_data
from data_preprocessing import preprocess_data
from model import build_lstm_model, train_model
from evaluate import evaluate_model

def main():
    ticker = "AAPL"
    start_date = "2010-01-01"
    end_date = "2023-01-01"
    sequence_length = 60
    epochs = 20
    batch_size = 32

    print("Fetching data...")
    df = fetch_stock_data(ticker, start_date, end_date)

    print("Preprocessing data...")
    x_train, y_train, x_test, y_test, scaler = preprocess_data(df, sequence_length)

    print("Building model...")
    model = build_lstm_model((x_train.shape[1], 1))

    print("Training model...")
    train_model(model, x_train, y_train, epochs, batch_size)

    print("Evaluating model...")
    evaluate_model(model, x_test, y_test, scaler)

if __name__ == "__main__":
    main()