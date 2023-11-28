import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-deep")

qlearn_mean = np.load("q_learn_mean-10_trials.npy")
qlearn_stdv = np.load("q_learn_err-10_trials.npy")

ne_mean = np.load("ne_learn_mean-10_trials.npy")
ne_stdv = np.load("ne_learn_err-10_trials.npy")

plt.plot(qlearn_mean, label="Q-learning")
plt.fill_between(np.arange(0, len(qlearn_mean)), qlearn_mean+qlearn_stdv, qlearn_mean-qlearn_stdv, alpha=0.4)
plt.plot(ne_mean, label="Neuro-Control")
plt.fill_between(np.arange(0, len(ne_mean)), ne_mean+ne_stdv, ne_mean-ne_stdv, alpha=0.4)
plt.xlabel("Epochs")
plt.ylabel("Grid World Training Reward")
plt.title("Static Grid World Training Performance")
plt.legend(loc="lower right")
plt.show()