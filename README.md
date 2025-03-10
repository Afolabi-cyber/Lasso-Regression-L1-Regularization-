# **Lasso Regression (L1 Regularization) from Scratch**

This project implements **Lasso Regression (L1 Regularization)** using **Gradient Descent**, built purely in Python without external libraries like NumPy or Scikit-learn.

---

## **1. Understanding Lasso Regression**
Lasso Regression is an extension of Linear Regression that adds an **L1 penalty** to the cost function to encourage sparsity and feature selection. The equation for prediction remains:

$$
y = b_0 + b_1X_1 + b_2X_2 + \dots + b_nX_n
$$

However, the cost function now includes an **L1 regularization term**:

$$
MSE_{lasso} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{n} |b_j|
$$



where:
- $y_i$ is the **actual value**.
- $\hat{y}_i$ is the **predicted value**.
- $n$ is the **total number of data points**.
- $\lambda$ (lambda) is the **regularization parameter** that controls the strength of the penalty.

The second term penalizes large coefficient values, **shrinking some to zero** (sparse model). Lasso Regression is particularly useful for **feature selection**, as it forces some coefficients to become exactly zero, reducing model complexity.

---

## **2. Gradient Descent Optimization**
The gradients for Lasso Regression differ from Ridge Regression due to the **absolute value** in the penalty term. The derivative of $|b_j|$ is not straightforward, leading to the following update rule:

$$
\frac{\partial MSE_{lasso}}{\partial b_j} = -\frac{2}{n} \sum_{i=1}^{n} X_{ij} (y_i - \hat{y}_i) + \lambda \cdot \text{sign}(b_j)
$$

where:
- **sign($b_j$)** is:
  - $+1$ if $b_j > 0$,
  - $-1$ if $b_j < 0$,
  - $0$ if $b_j = 0$.
- The **intercept** $b_0$ is **not regularized**.

Using these gradients, we update the parameters iteratively:

$$
b_j = b_j - \alpha \left(\frac{\partial MSE_{lasso}}{\partial b_j}\right)
$$

where $\alpha$ is the **learning rate**.

### **💡 Shrinkage Effect**
The **L1 penalty** causes some coefficients to **shrink to zero**, effectively selecting only the most important features.

---

## **3. Implementation Details**
### **Custom Lasso Regression Class**
We build a `LassoRegression` class with the following methods:
- `fit(X, y)`: Trains the model using gradient descent.
- `predict(X)`: Makes predictions using the trained model.
- **Regularization** is applied using the L1 penalty in the gradient update step.

---

## **📌 Summary**
✅ **Lasso Regression** adds L1 regularization, which encourages sparsity in features.  
✅ The **Mean Squared Error (MSE)** is minimized with an additional **absolute penalty term**.  
✅ **Gradient Descent** is used to update the parameters iteratively.  
✅ The **lambda ($\lambda$) parameter** controls the strength of regularization and feature selection.  
✅ The **L1 penalty** forces some coefficients to be **exactly zero**, leading to feature selection.  
✅ This implementation is built from scratch using **only core Python**.  

🚀 **Happy Coding!**

