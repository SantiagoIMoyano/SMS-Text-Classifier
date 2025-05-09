# SMS-Text-Classifier

## Repository Content

In this repository you’ll find the “SMS Text Classifier” project, a machine learning application that uses both dense neural networks and LSTM networks.
The project covers everything from obtaining and converting the raw data into formats the neural network can understand, to validating the associated tests.
You’ll notice the code is fully modularized—allowing you to train, evaluate, predict, and test independently—so you don’t have to run the entire program from start to finish each time. Additionally, the `notebooks` folder contains the complete project notebook linked to the “Machine Learning with Python” certification.

## Technologies Used

- Language: Python
- Framework: TensorFlow
- Libraries: Pandas, NumPy, Keras, Pytest

## Project Execution Notes

Note that to fully run this project using the `run_all.sh` script, you must open a Linux Bash terminal.

Feel free to run whichever commands are necessary to train, evaluate, predict, or run the tests without relying on the `run_all.sh` script. Since the project’s code is divided into separate components, you don’t need to execute the entire program at once to use the model for prediction or testing.

To run this project successfully, follow these steps:

1. Clone the repository using the command `git clone`.

2. Navigate to the project’s root directory and run `python -m venv venv` to create a virtual environment that isolates the project and ensures it runs correctly.

3. Finally, execute the command `sh run_all.sh`, which will start the program in its entirety.