"""
Reproducing https://github.com/karpathy/micrograd
"""

import math
from typing import Union


class Value:
    """
    A value can be constructed using a series of operations. Value will memorise 
    this series, and roll backwards when val._backward() is called
    """

    def __init__(
        self, 
        data: float,
        _children: tuple = (),
        _op: str | None = None,
        _label: str | None = None,
    ):
        self.data = data
        # gradient of objective function in respect to self
        self.grad: float = 0.0
        self._children = _children
        self._backward = lambda: None
        self._op = _op

    def __repr__(self):
        res = f'Value({self.data:+6.2f}'
        if self.grad is not None:
            res += f', grad={self.grad:+6.2f}'
        res += ')'
        return res
    
    def __str__(self):
        return self.__repr__()

    def __neg__(self):
        return self * -1

    def __add__(self, other: Union['Value', int, float]):
        other = Value(other) if isinstance(other, int | float) else other
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # The derivative of summation is f(a)+b /da = f(a)/da (passthrough)
            # 
            # We need to be mindful that expressions like `a = a + a` or `a += a + 1`
            # are effectively multiplications by 2 (`a = 2*a + ...`), so the derivative
            # should be 2, not 1 like in addition. To account for that, we initialised
            # our grad with 0.0, and accumulate it with += instead of rewriting with =
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other: Union[int, float, 'Value']):
        return self + other
    
    def __sub__(self, other: Union[int, float, 'Value']):
        return self + (-other)

    def __rsub__(self, other: Union[int, float, 'Value']):
        return other + (-self)
        
    def __mul__(self, other: Union[int, float, 'Value']):
        other = Value(other) if isinstance(other, int | float) else other
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            """
            The derivative of multiplication is just the other multiplier:
            (k * f(a)) /da = k * f(a)/da
            """
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def __rmul__(self, other: Union[int, float, 'Value']):
        return self * other

    def __truediv__(self, other: Union[int, float, 'Value']):
        """
        Same as multiplier a reversed value; and reverse is the same as power by -1
        """
        return self * other**-1
    
    def __rtruediv__(self, other: Union[int, float, 'Value']):
        return other * self**-1

    def __pow__(self, p: int | float):
        """
        The derivative of power function is:
            f(a)**p /da = p*(f(a)**(p-1)) * f(a)/da 
        where f(a) is just `self.data`
        """
        out = Value(self.data ** p, (self,), f'**{p}')

        def _backward():
            self.grad += out.grad * p * self.data**(p-1)
        out._backward = _backward
        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'relu')

        def _backward():
            self.grad += 0 if self.data < 0 else out.grad
        out._backward = _backward
        return out

    def tanh(self):
        out = Value(
            (math.exp(2*self.data) - 1) / (math.exp(2**self.data) + 1), 
            (self,), 
            'tanh',
        )

        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        """
        Will populate *.grad for all previous values. 
        E.g. following the forward pass for a chain of functions:
            a = Value(2.0)
            b = Value(-3.0)
            c = Value(0.5)
            f = a * b   # f(a) = a*b = -6.0
            g = f * c   # g(a) = (a*b)*c = -6.0 * 0.5 = 3.0
        *.data will be populated first: 
            a.data == +2.0
            b.data == -3.0
            c.data == +0.5
            f.data == -6.0
            g.data == -3.0
        And after calling `g.backward()`, *.grad will be populated as well:
            g.grad = 1.0  # dg/dg = 1
            f.grad = dg/df = (f*c) /df = 0.5
            a.grad = dg/da = dg/df * df/da = f.grad * df/da = 0.5 * (a*b) /da = 0.5 * b = 0.5 * -3 = -1.5
        """
        # The gradient calculation only depends on the value's data, and 
        # the _output_ values' gradient. So we only need to guarantee that by the time
        # gradient is calculated for a value, the outputs that use these value
        # already have their gradient calculated. To guarantee that, we need to
        # do a topological sorting on all values.
        topo = []
        visited = set()
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    build_topo(child)
                topo.append(node)
        build_topo(self)
        topo = topo[::-1]
        
        self.grad = 1.0  # Gradient of a function in respect to self is 1.0
        for node in topo:
            node._backward()
            if node._op:
                print(f'{node._op:<4} {node} <- {node._children}')


if __name__ == '__main__':
    def estimate_grad(forward_fn, *inputs, eps=0.0001):
        out1 = forward_fn(*inputs)
        for i in range(len(inputs)):
            inputs2 = list(inputs)
            inputs2[i] += eps
            out2 = forward_fn(*inputs2)
            print('Est grad:', ((out2 - out1) / eps).data)

    def forward(a):
        return a + a + a

    a = Value(2.0)
    h = forward(a)
    h.backward()
    print(f'Calculated: {a.grad=}')
    estimate_grad(forward, a)
    print()
        
    def forward(a):
        b = Value(-3.0)
        c = Value(10.0)
        d = Value(-2.0)
        f = a * b   # f(a) = a*b = -6.0
        g = f + c   # g(a) = (a*b)*c = -6.0 + 10.0 = 4.0
        h = g * d   # 4.0 * -2 = -8.0
        return h

    a = Value(2.0)
    h = forward(a)
    h.backward()
    print(f'Calculated: {a.grad=}')
    estimate_grad(forward, a)
    print()

    def forward(a, b):
        c = a + b                      # -2.0
        d = a * b + b**3               # -8.0 + 8.0 = 0.0
        c += c + 1
        c += 1 + c + (-a)              # -3.0 + 1 + -3.0 + 4.0 = -1.0 
        d += d * 2 + (b + a).relu()    # 0.0
        d += 3 * d + (b - a).relu()    # 6.0
        e = c - d                      # -1.0 - 6.0 = -7.0
        f = e**2                       # 49.0
        g = f / 2.0                    # 24.5
        g += 10.0 / f                  # 24.5 + 0.2040816327 = 24.7040816327
        return g

    b = Value(2.0)
    a = Value(-4.0)
    res = forward(a, b)
    print(f'Result: {res.data=:.4f}')  # prints 24.7041, the outcome of this forward pass
    res.backward()
    print(f'Est grad: {a.grad=:.4f}')  # prints 138.8338, i.e. the numerical value of dg/da
    print(f'Calculated grad: {b.grad=:.4f}')  # prints 138.8338, i.e. the numerical value of dg/da
    # print(f'{b.grad=:.4f}')  # prints 645.5773, i.e. the numerical value of dg/db
    estimate_grad(forward, a, b)
