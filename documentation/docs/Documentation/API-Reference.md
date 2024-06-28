<a id="nn.loss"></a>

# nn.loss

<a id="nn.loss.MeanSquaredError"></a>

## MeanSquaredError Objects

```python
class MeanSquaredError()
```

Class to compute the Mean Squared Error (MSE) and its gradient.

<a id="nn.loss.MeanSquaredError.forward"></a>

#### forward

```python
def forward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray
```

Compute the Mean Squared Error between true and predicted values.

**Arguments**:

- `y_true` _np.ndarray_ - True values.
- `y_pred` _np.ndarray_ - Predicted values.
  

**Returns**:

- `np.ndarray` - The mean squared error.
  

**Raises**:

- `ValueError` - If y_true and y_pred do not have the same shape.

<a id="nn.loss.MeanSquaredError.backward"></a>

#### backward

```python
def backward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray
```

Compute the gradient of the Mean Squared Error with respect to the predicted values.

**Arguments**:

- `y_true` _np.ndarray_ - True values.
- `y_pred` _np.ndarray_ - Predicted values.
  

**Returns**:

- `np.ndarray` - The gradient of the loss with respect to y_pred.

