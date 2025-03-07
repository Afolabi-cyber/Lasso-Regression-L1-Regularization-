# **Lasso Regression (L1 Regularization) from Scratch**  

This project implements **Lasso Regression** (L1 Regularization) using **Gradient Descent**, built purely in Python 

---

## **1. Understanding Lasso Regression**  

Lasso Regression is an extension of **Linear Regression** that adds an **L1 penalty** to the cost function to encourage sparsity and feature selection. The equation for prediction remains:  

\[
y = b_0 + b_1X_1 + b_2X_2 + \dots + b_nX_n
\]

However, the cost function now includes an L1 regularization term:

\[
MSE_{lasso} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{n} |b_j|
\]

where:  

- **yᵢ** is the actual value.  
- **ŷᵢ** is the predicted value.  
- **n** is the total number of data points.  
- **λ** (lambda) is the regularization parameter that controls the strength of the penalty.  
- The second term penalizes large coefficient values, driving some to zero (sparse model).  

Lasso Regression is useful when **feature selection** is important, as it forces some coefficients to become exactly **zero**.  

---

## **2. Gradient Descent Optimization**  

The gradients for Lasso Regression differ from Ridge Regression due to the **absolute value in the penalty term**. The derivative of **|bⱼ|** is not straightforward, leading to the following update rule:  

\[
\frac{\partial MSE_{lasso}}{\partial b_j} =
-\frac{2}{n} \sum_{i=1}^{n} X_{ij} (y_i - \hat{y}_i) + \lambda \cdot \text{sign}(b_j)
\]

where:

- **sign(bⱼ)** is **+1 if bⱼ > 0**, **-1 if bⱼ < 0**, and **0 if bⱼ = 0**.
- **b₀ (intercept)** is not regularized.

Using these gradients, we update the parameters iteratively:

\[
b_j = b_j - \alpha \left(\frac{\partial MSE_{lasso}}{\partial b_j}\right)
\]

where **α** is the learning rate.

---

## **3. Implementation Details**  

### **Custom Lasso Regression Class**  

We build a **LassoRegression** class with the following methods:

- **fit(X, y)**: Trains the model using gradient descent.  
- **predict(X)**: Makes predictions using the trained model.  
- **Regularization** is applied using the **L1 penalty** in the gradient update step.  

---

### **📌 Summary**  

✅ Lasso Regression adds **L1 regularization**, which encourages sparsity in features.  
✅ The **Mean Squared Error (MSE)** is minimized with an additional absolute penalty term.  
✅ **Gradient Descent** is used to update the parameters iteratively.  
✅ The **lambda (λ) parameter** controls the strength of regularization and feature selection.  
✅ This implementation is built **from scratch** using only core Python.  

🚀 **Happy Coding!**  

---

Let me know if you’d like any changes! 🎯
