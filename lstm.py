import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)


#----> Forget gate weights
Wf_x, Wf_h, bf = 0.5, 0.6, 0.1

#---->Input gate weights
Wi_x, Wi_h, bi = 0.7, 0.8, 0.1

#----->Candidate cell weights
Wc_x, Wc_h, bc = 0.4, 0.9, 0.1

#-----> Output gate weights
Wo_x, Wo_h, bo = 0.6, 0.5, 0.1

#------> Output layer 
Wy, by_out = 1.0, 0.0

# ------> Initial states
h_prev = 0.0   
c_prev = 0.0   

inputs = [1, 2, 3, 4]

print("=" * 55)
print("   LSTM Forward Pass — Numerical Example")
print("=" * 55)

for t, x in enumerate(inputs, start=1):
    print(f"\n{'─'*50}")
    print(f" Time Step t={t},  Input x={x}")
    print(f"{'─'*50}")
    print(f"  h_prev = {h_prev:.4f},  c_prev = {c_prev:.4f}")

    # --> Forget gate
    f_t = sigmoid(Wf_x * x + Wf_h * h_prev + bf)
    print(f"\n  1) Forget gate:")
    print(f"     f = σ({Wf_x}×{x} + {Wf_h}×{h_prev:.4f} + {bf})")
    print(f"     f = σ({Wf_x*x + Wf_h*h_prev + bf:.4f}) = {f_t:.4f}")

    # --> Input gate
    i_t = sigmoid(Wi_x * x + Wi_h * h_prev + bi)
    print(f"\n  2) Input gate:")
    print(f"     i = σ({Wi_x}×{x} + {Wi_h}×{h_prev:.4f} + {bi})")
    print(f"     i = σ({Wi_x*x + Wi_h*h_prev + bi:.4f}) = {i_t:.4f}")

    # --> Candidate cell state
    c_tilde = tanh(Wc_x * x + Wc_h * h_prev + bc)
    print(f"\n  3) Candidate cell state:")
    print(f"     c̃ = tanh({Wc_x}×{x} + {Wc_h}×{h_prev:.4f} + {bc})")
    print(f"     c̃ = tanh({Wc_x*x + Wc_h*h_prev + bc:.4f}) = {c_tilde:.4f}")

    # --> Cell state update
    c_t = f_t * c_prev + i_t * c_tilde
    print(f"\n  4) Cell state update:")
    print(f"     c = f×c_prev + i×c̃")
    print(f"     c = {f_t:.4f}×{c_prev:.4f} + {i_t:.4f}×{c_tilde:.4f} = {c_t:.4f}")

    # -Output gate
    o_t = sigmoid(Wo_x * x + Wo_h * h_prev + bo)
    print(f"\n  5) Output gate:")
    print(f"     o = σ({Wo_x}×{x} + {Wo_h}×{h_prev:.4f} + {bo})")
    print(f"     o = σ({Wo_x*x + Wo_h*h_prev + bo:.4f}) = {o_t:.4f}")

    # -->Hidden state update
    h_t = o_t * tanh(c_t)
    print(f"\n  6) Hidden state update:")
    print(f"     h = o × tanh(c)")
    print(f"     h = {o_t:.4f} × tanh({c_t:.4f}) = {h_t:.4f}")

    
    h_prev = h_t
    c_prev = c_t

y_hat = Wy * h_prev + by_out
print(f"\n{'='*55}")
print(f" Step 3: Predict Next Value")
print(f"{'='*55}")
print(f"  ŷ = Wy × h_final + by")
print(f"  ŷ = {Wy} × {h_prev:.4f} + {by_out}")
print(f"\n Final Prediction: ŷ = {y_hat:.4f}")
print(f"     (Expected ≈ 4,  PDF result: 3.8)")
print("=" * 55)