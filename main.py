import math
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Title of web-interface
st.title("Analysis of IVP")
st.header("y' = (1 + y/x)𝑙𝑛((x + y)/x) + y/x")
st.subheader("Please, input arguments below:")

class GUI(object):
    def input_N(self, text):
        n = st.number_input(label=text,
                            min_value=1, max_value=1000, value=10)
        return n

    def input_y0(self, text):
        y_0 = st.number_input(label=text,
                              value=2.0)
        return y_0

    def input_x0(self, text):
        x_0 = st.number_input(label=text,
                              value=1.0)
        return x_0

    def input_X(self, text):
        x = st.number_input(label=text,
                            value=6.0, max_value=646.0)
        return x

    def input_Ns(self, text1, text2):
        n0 = st.number_input(label=text1,
                             min_value=1, max_value=999, value=10)
        N = st.number_input(label=text2,
                            min_value=1, max_value=999, value=50)
        return n0, N

    def plot_graph1(self, xArr, emYArr, ieYArr, rkYArr, asYArr, columnText, titleText):
        # Create figure with secondary y-axis
        fig = make_subplots()

        # Add lines for each graph
        fig.add_trace(go.Scatter(x=xArr, y=emYArr, name=columnText[0]))
        fig.add_trace(go.Scatter(x=xArr, y=ieYArr, name=columnText[1]))
        fig.add_trace(go.Scatter(x=xArr, y=rkYArr, name=columnText[2]))
        fig.add_trace(go.Scatter(x=xArr, y=asYArr, name=columnText[3]))

        # Add figure title
        fig.update_layout(title_text=titleText)

        # Set x-axis title
        fig.update_xaxes(title_text="X")

        # Set y-axes titles
        fig.update_yaxes(title_text="Y")

        st.write(fig)

    def plot_graph2(self, xArr, eulerM, impEuler, runKut, columnText, titleText, xAxisText, yAxisText):
        # Create figure with secondary y-axis
        fig = make_subplots()

        # Add lines for each graph
        fig.add_trace(go.Scatter(x=xArr, y=eulerM, name=columnText[0]))
        fig.add_trace(go.Scatter(x=xArr, y=impEuler, name=columnText[1]))
        fig.add_trace(go.Scatter(x=xArr, y=runKut, name=columnText[2]))

        # Add figure title
        fig.update_layout(title_text=titleText)

        # Set x-axis title
        fig.update_xaxes(title_text=xAxisText)

        # Set y-axes titles
        fig.update_yaxes(title_text=yAxisText)

        st.write(fig)

    def call_error(self, text):
        st.error(text)



class Grid(GUI):
    myInput = GUI()
    N = myInput.input_N("Number of steps. N: ")
    y0 = myInput.input_y0("Initial value for y_0: ")
    x0 = myInput.input_x0("Initial value for x_0: ")
    X = myInput.input_X("Initial value for X: ")
    h = (X - x0)/N
    xArr = [x0]
    def fill_x_array(self):
        for i in range(1, self.N+1):
            self.xArr.append(self.xArr[i-1]+self.h)


class AnalyticalSolution(Grid):
    grid = Grid()
    c = 0
    asYArr = [grid.y0]

    def compute_const(self):
        x0 = self.grid.x0
        y0 = self.grid.y0
        if x0 == 0:
            self.c = 1
        elif x0*y0 < 0 and abs(y0) >= abs(x0):
            return 0
        else:
            self.c = math.log(1 + grid.y0 / grid.x0) / grid.x0
        return 1

    def f_ans(self, x):
        # Exact solution gained through computations
        return x * (math.exp(self.c * x) - 1)

    def fill_array(self):
        x_i = self.grid.x0
        h = self.grid.h
        for i in range(1, self.grid.N + 1):
            x_i += h
            self.asYArr.append(self.f_ans(x_i))



class NumericalMethod(Grid):
    grid = Grid()
    flag = 0
    def f(self, x, y):
        if 1 + y/x < 0:
            self.flag = 1
            return (1 + y / x) * math.log(1 + abs(y / x)) + y / x
        return (1 + y / x) * math.log(1 + y / x) + y / x


class EulerMethod(NumericalMethod):
    num = NumericalMethod()
    emYArr = [num.grid.y0]

    def y_method(self, x, y, h):
        return y + h * self.num.f(x, y)

    def fill_array(self):
        x_i = self.num.grid.x0
        h = self.num.grid.h
        for i in range(1, self.num.grid.N + 1):
            y_i = self.emYArr[i - 1]
            # Euler formula
            tmpY = self.y_method(x_i, y_i, h)
            # getting x_(i+1) after finding value of y_i through x_i
            x_i += h

            self.emYArr.append(tmpY)


class ImprovedEuler(NumericalMethod):
    num = NumericalMethod()
    ieYArr = [num.grid.y0]

    def y_method(self, x, y, h):
        return y + h * self.num.f(x + h/2, y + h/2 * self.num.f(x, y))

    def fill_array(self):
        x_i = self.num.grid.x0
        h = self.num.grid.h
        for i in range(1, self.num.grid.N + 1):
            y_i = self.ieYArr[i - 1]
            # Improved Euler formula
            tmpY = self.y_method(x_i, y_i, h)
            # getting x_(i+1) after finding value of y_i through x_i
            x_i += h

            self.ieYArr.append(tmpY)


class RungeKutta(NumericalMethod):
    num = NumericalMethod()
    rkYArr = [num.grid.y0]

    def k1(self, x, y):
        return self.num.f(x, y)

    def k2(self, x, y, h):
        return self.num.f(x + h / 2, y + h * self.k1(x, y) / 2)

    def k3(self, x, y, h):
        return self.num.f(x + h / 2, y + h * self.k2(x, y, h) / 2)

    def k4(self, x, y, h):
        return self.num.f(x + h, y + h * self.k3(x, y, h))

    def y_method(self, x, y, h):
        return y + h / 6 * (self.k1(x, y) +
                              2 * self.k2(x, y, h) +
                              2 * self.k3(x, y, h) +
                              self.k4(x, y, h))

    def fill_array(self):
        x_i = self.num.grid.x0
        h = self.num.grid.h
        for i in range(1, self.num.grid.N + 1):
            y_i = self.rkYArr[i - 1]
            # Improved Runge-Kutta formula
            tmpY = self.y_method(x_i, y_i, h)
            # getting x_(i+1) after finding value of y_i through x_i
            x_i += h

            self.rkYArr.append(tmpY)

# Function to solve everything
class Solve(object):
    def __init__(self, grid):
        grid.fill_x_array()

        gui = GUI()

        analSol = AnalyticalSolution()
        if analSol.compute_const() == 0:
            gui.call_error("You wrote invalid x0 or y0")
            return

        eulerM = EulerMethod()
        impEuler = ImprovedEuler()
        runKut = RungeKutta()

        # Some calculations to plot needed graphs
        X = grid.xArr
        epsilon = 0.1
        # Point of discontinuity error
        for x in X:
            if abs(x) < epsilon:
                gui.call_error(
                    "One of the x_i reaching function's break point, so methods become not aplicaple. Please, enter another N, y0, x0, or X")
                return

        Y_exact = analSol.asYArr
        h = grid.h
        # Very big step error
        if h > 1:
            gui.call_error("Your N is too small, so step h > 1")
            return
        # Filling arrays of methods and anal. solution
        analSol.fill_array()
        eulerM.fill_array()
        impEuler.fill_array()
        runKut.fill_array()
        # Math error (flag defines that math error happened somewhere)
        if runKut.num.flag == 1:
            grid.myInput.call_error("Please, enter another N, y0, x0, or X. Function f(x, y) has math error (negative logarithm)")
            return
        columnText = ["Eulers M.", "Improved Euler's M.", "Runge-Kutta M.", "Analytical Solution"]
        # Plotting first graph
        gui.plot_graph1(X, eulerM.emYArr, impEuler.ieYArr, runKut.rkYArr, analSol.asYArr, columnText,
                        "Solution of IVP using different approximations")

        # Some calculations on computing LTEs
        emLTEs = [0]
        ieLTEs = [0]
        rkLTEs = [0]
        for i in range(1, grid.N+1):
            emLTEs.append(abs(Y_exact[i] - eulerM.y_method(X[i-1], Y_exact[i-1], h)))
            ieLTEs.append(abs(Y_exact[i] - impEuler.y_method(X[i-1], Y_exact[i-1], h)))
            rkLTEs.append(abs(Y_exact[i] - runKut.y_method(X[i-1], Y_exact[i-1], h)))

        # Plotting second graph
        gui.plot_graph2(X, emLTEs, ieLTEs, rkLTEs, columnText,
                        "Graph of local truncation errors (LTE)", "X", "LTE")

        # Some calculations before plotting third graph
        emMaxLTEs = []
        ieMaxLTEs = []
        rkMaxLTEs = []
        n0, N = gui.input_Ns("Initial number of steps. n0: ", "Number of steps. N: ")
        # Handling wrong input of n0 > N
        if n0 > N:
            gui.call_error("Your n0 larger than N")
            return

        nArr = []
        for i in range(n0, N+1):
            nArr.append(i)

        for nTmp in nArr:
            h = (grid.X-grid.x0)/nTmp
            if h > 1:
                gui.call_error("Your n0 is too small, so step h > 1")
                return
            x = grid.x0
            yArr = [grid.y0]

            maxEMlte = 0
            maxIElte = 0
            maxRKlte = 0
            for i in range(1, nTmp):
                x += h
                yArr.append(analSol.f_ans(x))
                maxEMlte = max(maxEMlte, abs(yArr[i] - eulerM.y_method(x-h, yArr[i-1], h)))
                maxIElte = max(maxIElte, abs(yArr[i] - impEuler.y_method(x-h, yArr[i-1], h)))
                maxRKlte = max(maxRKlte, abs(yArr[i] - runKut.y_method(x-h, yArr[i-1], h)))

            emMaxLTEs.append(maxEMlte)
            ieMaxLTEs.append(maxIElte)
            rkMaxLTEs.append(maxRKlte)

        # Plotting third graph
        gui.plot_graph2(nArr, emMaxLTEs, ieMaxLTEs, rkMaxLTEs, columnText,
                        "Graph of max LTE at n_i-th step from n0 to N", "N", "GTE")

# Main method
grid = Grid()
solve = Solve(grid)


