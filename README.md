# nn-from-scratch

My personal repository following Andrej Karpathy's **Neural Networks: Zero to Hero** lectures.

## Tiny Autograd Engine (Micrograd)

A tiny Autograd engine (with a bite! 😊). Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural network library on top of it with a PyTorch-like API.

Both the autograd engine and the neural net library are tiny (~100 and ~50 lines of code respectively).

---

### History / Progress

**April 22, 2026**

**File:** [`andrej_lectures/micrograd_from_scratch.ipynb`](https://github.com/mithxr/nn-from-scratch/blob/main/andrej_lectures/micrograd_from_scratch.ipynb)

**What it covers:**

1. Intuitive understanding of derivatives, slope, and differentiability
2. Practical examples to understand what the derivative of a loss function w.r.t a variable means
3. Creation of a `Value` data structure to represent mathematical expressions for neural networks
4. Understanding the forward pass and laying the groundwork for the backward pass (gradients)
5. Visual representation of the computation graph using Graphviz (`micrograd_graph.png`)

---

### Project Structure

```bash
nn-from-scratch/
├── andrej_lectures/
│   └── micrograd_from_scratch.ipynb
├── micrograd_graph.png
├── .gitignore
└── README.md
