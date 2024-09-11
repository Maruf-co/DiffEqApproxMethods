# Differential Equations Approximation Methods

## Overview
This project provides a set of tools for approximating solutions to differential equations using various numerical methods. The implementation is written in Python and leverages libraries such as Streamlit for interactive web applications, Plotly for data visualization, and built-in mathematical functions from the math library.

## Features
- Interactive user interface for inputting differential equations and parameters.
- Visualization of the approximated solutions using Plotly.
- Implements various numerical methods for solving ordinary differential equations (ODEs).

## Dependencies
This project requires the following Python libraries:

- math: Standard library for mathematical functions.
- streamlit: For creating the web interface.
- plotly: For generating interactive plots and visualizations.

You can install the required dependencies using pip:

pip install streamlit plotly


## Getting Started

### Clone the Repository
Clone this repository to your local machine:

git clone https://github.com/yourusername/differential-equations-approximation.git
cd differential-equations-approximation


### Running the Application
To run the web application, execute the following command in your terminal:

streamlit run app.py


### Usage
1. Open your browser and navigate to http://localhost:8501.
2. Input the differential equation you want to approximate.
3. Set the parameters for your chosen numerical method.
4. Visualize the results using interactive plots.

## Methods Implemented
- Euler's Method
- Improved Euler Method (Heun's Method)
- Runge-Kutta Methods (2nd and 4th Order)

## Example
Here’s an example of how to use the application:

1. Enter a first-order ODE in the format dy/dx = f(x, y).
2. Specify the initial conditions.
3. Select the numerical method and the interval for the approximation.
4. Click "Submit" to see the plot of the approximation.

## Contributing
Contributions to improve the functionality and features of this project are welcome! If you have suggestions or improvements, feel free to fork the repository and create a pull request, or open an issue.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Streamlit (https://streamlit.io/) for providing an easy way to create web applications.
- Plotly (https://plotly.com/python/) for offering powerful graphing tools in Python.

Feel free to reach out if you have any questions or need assistance with the project!
