#%%
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#%%
class Perceptron:
    def __init__(self, seed=0, input_size=2, learning_rate=0.01, epochs=100):
        self.seed = seed
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.input_size = input_size
        self._init_weights()

    def _init_weights(self):
        rng = np.random.default_rng(self.seed)
        ### START CODE HERE ###
        ### TODO: Initialize weights with small Gaussian noise using rng.normal
        self.weights = rng.normal(-1,1.0, size=self.input_size + 1)# +1 por conta do bias
        ### END CODE HERE ###

    def activation(self, x):
        ### START CODE HERE ###
        ### TODO: Implement the step activation function
        return np.where(x >= 0, 1,-1) # se for maiorigual a 0, vira 1, se nao, -1
        ### END CODE HERE ###

    def predict(self, X):
        ### START CODE HERE ###
        ### TODO: Add bias term, compute dot product with weights, apply activation
        # preenche a coluna de cada linha das entradas xn com o numero 1, para ser o bias
        #np.column_stack -> transforma um array de n 1s em uma coluna e insere na matriz X
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        product = np.dot(X_with_bias, self.weights)
        return self.activation(product)
        ### END CODE HERE ###

    def fit(self, X, y):
        ### START CODE HERE ###
        ### TODO: Implement the perceptron learning algorithm
        for epoch in range(self.epochs):
            #inicio as entradas
            x_with_bias = np.insert( X, 0, 1, axis=1)
            for input, true_label in zip(x_with_bias, y):
                #para cada entrada e saida esperada
                #input = array = serie de inputs, cada caso = x_with_bias
                product = np.dot(input, self.weights) #produto de matriz entre entradas x1,x2,1 e w1,w2,w3
                output = self.activation(product) # output = a
                if output != true_label: #metrica => a !+ y
                    loss = (true_label - output)/2 # loss = (y-a)/2
                    update = self.learning_rate * loss * input # update = N * L(a,y) * xn
                    self.weights = self.weights + update # weights = wn + L(a,y)
                    
        
        ### END CODE HERE ###

#%%
def generate_halfmoon(seed = 0,
                      n_samples=1000,
                      rad_min=50,
                      rad_max=100,
                      ang_min=45,
                      ang_max=220,
                      dist_type='normal',
                      rad_std=10,
                      ang_std=30.0,
                      x0=0.0,
                      y0=0.0,
                      return_as_array=False):
    """
    Generate 2D data in a halfmoon shape using polar coordinate sampling.

    Parameters:
        n_samples (int): Number of points to generate.
        rad_min (float): Minimum radius.
        rad_max (float): Maximum radius.
        ang_min (float): Minimum angle in degrees.
        ang_max (float): Maximum angle in degrees.
        dist_type (str): 'uniform' or 'normal'.
        rad_std (float): Std dev for radius (used if dist_type='normal').
        ang_std (float): Std dev for angle (used if dist_type='normal').
        x0 (float): X offset of the moon.
        y0 (float): Y offset of the moon.
        return_as_array (bool): If True, returns Nx2 array. Else, returns (x, y) tuple.

    Returns:
        np.ndarray or tuple of np.ndarray: Data points (x, y).
    """
    np.random.seed(seed)

    dist_type = dist_type.lower()
    if dist_type not in ['uniform', 'normal']:
        raise ValueError("dist_type must be either 'uniform' or 'normal'.")

    if dist_type == 'uniform':
        r = np.random.uniform(rad_min, rad_max, size=n_samples)
        t = np.random.uniform(ang_min, ang_max, size=n_samples)
    else:  # normal
        r_mean = (rad_max + rad_min) / 2.0
        t_mean = (ang_max + ang_min) / 2.0
        r = np.random.normal(loc=r_mean, scale=rad_std, size=n_samples)
        t = np.random.normal(loc=t_mean, scale=ang_std, size=n_samples)
        r = np.clip(r, a_min=0, a_max=None)  # avoid negative radii

    x = x0 + r * np.cos(np.deg2rad(t))
    y = y0 + r * np.sin(np.deg2rad(t))

    if return_as_array:
        return np.vstack((x, y)).T
    else:
        return x.reshape(-1, 1), y.reshape(-1, 1)

def generate_halfmoon_data(n_samples=1000):
    x11, x12 = generate_halfmoon(n_samples=n_samples//2, dist_type='normal', return_as_array=False, x0=0,   y0=0, ang_min=0,    ang_max=180)
    x21, x22 = generate_halfmoon(n_samples=n_samples//2, dist_type='normal', return_as_array=False, x0=75, y0=80, ang_min=-180, ang_max=0)

    X1 = np.hstack([x11, x12])
    X2 = np.hstack([x21, x22])
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n_samples//2), -1*np.ones(n_samples//2)])
    return X, y

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, preds, alpha=0.3, levels=[-1, 0, 1], colors=['red', 'blue'])
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k', s=20)
    plt.title("Perceptron Decision Boundary on Halfmoon Data")
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def main():
    X, y = generate_halfmoon_data(n_samples=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Perceptron(epochs=100)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    acc = np.mean(predictions == y_test)
    print(f"Test Accuracy: {acc:.2f}")
    plot_decision_boundary(model, X, y)

if __name__ == "__main__":
    main()
