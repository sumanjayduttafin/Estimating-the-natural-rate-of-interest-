
# Arbitrage-Free Nelson–Siegel (AFNS) Model  
## Descriptive replication guide for the estimation code

This document explains, in plain but precise language, what the AFNS code is doing, why each step is needed, and how the implementation maps to the Christensen, Diebold, and Rudebusch (2011) paper.

The goal of the code is to estimate a **three-factor arbitrage-free term structure model** for zero-coupon yields, extract the latent factors, and decompose fitted yields into:
1. an **expectations component** (the average expected future short rate), and
2. a **term premium**.

The implementation follows the paper’s AFNS idea: keep the familiar **Nelson–Siegel** loading structure, but add the **no-arbitrage adjustment** implied by affine bond pricing.

---

## 1. What the model is trying to represent

A yield curve summarizes interest rates across maturities. For each date \(t\) and maturity \(\tau\), the model explains the observed yield \(y_t(\tau)\) using three latent factors:

- **Level**: the overall height of the curve
- **Slope**: the steepness of the short end
- **Curvature**: the hump in the middle of the curve

In Nelson–Siegel form, the yield is written as a linear combination of these factors with maturity-dependent loadings. AFNS keeps that structure but imposes the restrictions needed so that the implied bond prices are **arbitrage-free**.

---

## 2. Mathematical structure of the model

### 2.1 Latent factor vector

The hidden state is a 3-dimensional vector:

\[
X_t = 
\begin{bmatrix}
L_t \\
S_t \\
C_t
\end{bmatrix}
\]

where:

- \(L_t\): level factor
- \(S_t\): slope factor
- \(C_t\): curvature factor

These are not directly observed. The model infers them from the cross-section of yields.

---

### 2.2 Observation equation: yields as a function of the latent factors

For each maturity \(\tau\), the fitted yield is

\[
\hat y_t(\tau) = \Lambda(\tau)^\top X_t + d(\tau)
\]

where:

- \(\Lambda(\tau)\) is the Nelson–Siegel loading vector
- \(d(\tau)\) is the yield adjustment term required by no-arbitrage

Expanding the Nelson–Siegel part gives

\[
\hat y_t(\tau)
=
L_t
+
S_t \frac{1-e^{-\lambda \tau}}{\lambda \tau}
+
C_t \left(
\frac{1-e^{-\lambda \tau}}{\lambda \tau} - e^{-\lambda \tau}
\right)
+
d(\tau)
\]

The parameter \(\lambda\) controls the decay rate of the loadings and therefore the location of the curve’s hump.

---

### 2.3 State equation: how the factors evolve over time

The latent factors evolve according to a Gaussian mean-reverting process. In discrete monthly form, the code uses

\[
X_t = c + \Phi X_{t-1} + \eta_t,
\qquad
\eta_t \sim \mathcal{N}(0,Q)
\]

with:

- \(\Phi\): persistence matrix
- \(c\): intercept vector
- \(Q\): covariance matrix of state shocks

This is the discrete-time version of an Ornstein–Uhlenbeck process, which is standard in Gaussian affine term structure models.

---

### 2.4 Measurement noise

Observed yields are not perfectly equal to model-implied yields. The code therefore adds a measurement error:

\[
y_t(\tau) = \hat y_t(\tau) + \varepsilon_t(\tau),
\qquad
\varepsilon_t(\tau) \sim \mathcal{N}(0,H)
\]

This makes the estimation more realistic and stabilizes the filter.

---

## 3. Connection to Christensen, Diebold, and Rudebusch (2011)

The paper derives an **arbitrage-free Nelson–Siegel model** by embedding Nelson–Siegel yield loadings inside a continuous-time affine bond-pricing framework. The key result is that yields take the form:

\[
y(t,T)
=
X_t^{(1)}
+
\frac{1-e^{-\lambda(T-t)}}{\lambda(T-t)} X_t^{(2)}
+
\left[
\frac{1-e^{-\lambda(T-t)}}{\lambda(T-t)} - e^{-\lambda(T-t)}
\right] X_t^{(3)}
-
\frac{A(t,T)}{T-t}
\]

This is the paper’s central AFNS yield equation. The code implements the same idea in discrete time:

- the Nelson–Siegel loading matrix corresponds to the three loading terms above,
- the function `yield_adj(...)` approximates the \(-A(t,T)/(T-t)\) correction,
- the state equation is the discrete-time analog of the factor dynamics used in the paper.

In other words, the code is a practical implementation of the paper’s AFNS structure.

---

## 4. Step-by-step description of the code

---

### 4.1 `load_data(path)`

#### Purpose
This function loads the yield curve dataset and prepares it for estimation.

#### What it does
1. Reads the Excel file.
2. Uses the first column as the date index.
3. Converts the dates to a `DatetimeIndex`.
4. Converts yields from percentages into decimals.
5. Resamples the data to **monthly frequency** using the last observation in each month.
6. Converts maturity labels such as `"3m"` into numerical maturities in **years**.
7. Returns:
   - a DataFrame of monthly yields,
   - a maturity grid in years.

#### Why this is needed
The model is estimated in monthly time steps and requires maturities in years so that the Nelson–Siegel loading formulas and the no-arbitrage term can be evaluated correctly.

---

### 4.2 `ns_loadings(tau, lam)`

#### Purpose
This function constructs the Nelson–Siegel loading matrix for one or more maturities.

#### Mathematical form
For each maturity \(\tau\), the three loadings are:

\[
1,\quad
\frac{1-e^{-\lambda \tau}}{\lambda \tau},\quad
\frac{1-e^{-\lambda \tau}}{\lambda \tau} - e^{-\lambda \tau}
\]

#### Interpretation
- The first loading is always 1, so the level factor affects all maturities equally.
- The second loading declines slowly with maturity, so the slope factor mostly affects short maturities.
- The third loading creates a hump-shaped response, so the curvature factor matters most at intermediate maturities.

#### Why this matters
This is the core Nelson–Siegel structure. It is what makes the model easy to interpret and empirically stable.

---

### 4.3 `yield_adj(tau, lam, sigma)`

#### Purpose
This function computes the yield adjustment term \(-A(\tau)/\tau\) implied by the affine no-arbitrage structure.

#### Mathematical role
The AFNS model does not let yields be just a linear factor fit. The bond-pricing equations produce an additional maturity-dependent correction:

\[
d(\tau) = -\frac{A(\tau)}{\tau}
\]

The function evaluates this adjustment numerically by integrating the volatility-adjusted bond-loadings over maturity.

#### Economic meaning
This term captures the convexity and pricing effects that arise when bond prices must satisfy no-arbitrage restrictions. It is the difference between a purely statistical Nelson–Siegel fit and an arbitrage-free Nelson–Siegel fit.

#### Why it matters
Without this term, the model would resemble standard dynamic Nelson–Siegel. With it, the model becomes AFNS.

---

### 4.4 `build_matrices(params, grid, dt=1/12)`

#### Purpose
This is the model-construction function. It turns the parameter vector into the matrices needed for the Kalman filter.

#### Inputs
The parameter vector contains:

- \(\lambda\): Nelson–Siegel decay parameter
- \(\kappa\): mean-reversion speeds of the three factors
- \(\theta\): long-run factor means
- \(\sigma\): factor volatilities
- \(\sigma_{\text{meas}}\): measurement error standard deviation

#### What it constructs

##### (a) Loadings matrix
\[
\Lambda = \texttt{ns\_loadings(grid, lam)}
\]

This maps the latent factors into yields.

##### (b) Yield adjustment
\[
d = \texttt{yield\_adj(grid, lam, sigma)}
\]

This is the maturity-specific no-arbitrage correction.

##### (c) State transition matrix
\[
\Phi = \operatorname{diag}(e^{-\kappa_i \Delta t})
\]

This controls how persistent each factor is from one month to the next.

##### (d) State intercept
\[
c = (I-\Phi)\theta
\]

This ensures the state process mean-reverts toward \(\theta\).

##### (e) State covariance
\[
Q = \operatorname{diag}\left(
\frac{\sigma_i^2(1-\phi_i^2)}{2\kappa_i}
\right)
\]

This is the exact discrete-time variance implied by the continuous-time OU process.

##### (f) Measurement covariance
\[
H = \sigma_{\text{meas}}^2 I
\]

This assumes independent observation noise across maturities with common variance.

#### Why this matters
This function defines the entire state-space system used for estimation.

---

### 4.5 `kalman_filter(y_obs, Lambda, d, Phi, c, Q, H)`

#### Purpose
This function applies the Kalman filter to estimate the hidden factors and evaluate the likelihood.

#### What the Kalman filter does
At each date:

1. **Predict** the next state using the model.
2. **Compare** predicted yields to observed yields.
3. **Update** the state estimate using the prediction error.
4. **Accumulate** the log-likelihood.

#### Key equations

##### Prediction step
\[
X_{t|t-1} = \Phi X_{t-1|t-1} + c
\]

##### Innovation
\[
\nu_t = y_t - (\Lambda X_{t|t-1} + d)
\]

##### Innovation covariance
\[
S_t = \Lambda P_{t|t-1}\Lambda^\top + H
\]

##### Update step
\[
X_{t|t} = X_{t|t-1} + K_t \nu_t
\]

where \(K_t\) is the Kalman gain.

#### Why this matters
The factors are latent, so the Kalman filter is the main tool that extracts them from observed yields.

---

### 4.6 `rts_smoother(Xf, Pf, Xp, Pp, Phi)`

#### Purpose
This function performs the Rauch–Tung–Striebel backward smoother.

#### What it does
The Kalman filter gives real-time estimates using only information available up to each date. The smoother improves on this by using the full sample:

- it starts from the filtered estimates,
- then runs backward through time,
- and revises each state estimate using future observations.

#### Why this matters
The smoothed states are usually the best estimates of the latent level, slope, and curvature series.

---

### 4.7 `neg_loglik(params, y_obs, grid)`

#### Purpose
This function computes the negative log-likelihood for a given parameter vector.

#### What it does
1. Builds the state-space matrices.
2. Runs the Kalman filter.
3. Returns the negative of the log-likelihood.

#### Why this matters
Optimization routines minimize functions, so the code uses negative log-likelihood as the objective for estimation.

---

### 4.8 `estimate_model(y_obs, grid)`

#### Purpose
This is the maximum-likelihood estimation routine.

#### What it does
1. Sets an initial parameter guess.
2. Computes the initial likelihood.
3. Uses the L-BFGS-B optimizer to minimize the negative log-likelihood.
4. Returns the estimated parameters.

#### Why this matters
This is the function that actually fits the AFNS model to data.

#### Output
It produces estimates for:
- \(\lambda\)
- \(\kappa_L, \kappa_S, \kappa_C\)
- \(\theta_L, \theta_S, \theta_C\)
- \(\sigma_L, \sigma_S, \sigma_C\)
- \(\sigma_{\text{meas}}\)

These estimates determine the fitted yield curve and the latent factor dynamics.

---

### 4.9 `_print_params(params)`

#### Purpose
This helper function prints the estimated parameters in a human-readable form.

#### Why it matters
The raw parameter vector includes log-transformed values for positive parameters. This function converts them back into economically meaningful quantities.

---

### 4.10 `decompose_yields(...)`

#### Purpose
This function decomposes fitted yields into expectations and term premia.

#### Main idea
For a given maturity, the fitted yield is separated as:

\[
\text{Fitted yield} = \text{Expectations component} + \text{Term premium}
\]

#### Step 1: Fitted yield
\[
\hat y_t(\tau) = \Lambda(\tau)^\top X_t + d(\tau)
\]

#### Step 2: Expectations component
The model computes the average expected short rate over the maturity horizon.

The short rate in this AFNS implementation is:

\[
r_t = L_t + S_t
\]

So the expectations component is the average expected path of that short rate under the physical measure.

#### Step 3: Term premium
The term premium is the residual:

\[
TP_t(\tau) = \hat y_t(\tau) - EH_t(\tau)
\]

#### Why this matters
This decomposition gives the model economic meaning. It helps distinguish:
- expected policy-rate movements,
- from compensation for interest-rate risk.

---

## 5. Estimation workflow in practical terms

A researcher reproducing the model would follow these steps:

### Step 1: Collect and clean yield data
- Use zero-coupon yields at several maturities.
- Convert to decimals.
- Align them to monthly frequency.
- Ensure maturities are expressed in years.

### Step 2: Choose the model structure
- Use three latent factors: level, slope, curvature.
- Decide whether to use a diagonal/independent-factor structure, as in this implementation.

### Step 3: Set initial parameters
- Provide starting values for \(\lambda\), \(\kappa\), \(\theta\), \(\sigma\), and measurement noise.

### Step 4: Build the state-space system
- Compute the loading matrix.
- Compute the yield adjustment term.
- Construct the transition and observation matrices.

### Step 5: Evaluate the likelihood
- Run the Kalman filter.
- Sum the prediction-error likelihood across dates.

### Step 6: Estimate by maximum likelihood
- Optimize the parameter vector using a numerical optimizer.

### Step 7: Smooth the latent factors
- Use the smoother to obtain full-sample estimates of the hidden level, slope, and curvature series.

### Step 8: Decompose yields
- Compute fitted yields.
- Compute expected short rates.
- Recover term premia.

### Step 9: Plot and interpret results
- Inspect fitted-versus-observed yields.
- Examine factor paths.
- Review the term premium dynamics.

---

## 6. Relationship to the paper’s equations

The main mappings are:

### Paper equation for Nelson–Siegel yields
\[
y(t,T) =
X_t^{(1)}
+
\frac{1-e^{-\lambda(T-t)}}{\lambda(T-t)} X_t^{(2)}
+
\left[
\frac{1-e^{-\lambda(T-t)}}{\lambda(T-t)} - e^{-\lambda(T-t)}
\right] X_t^{(3)}
-
\frac{A(t,T)}{T-t}
\]

This corresponds directly to:

- `ns_loadings(...)`
- `yield_adj(...)`

### Paper’s state dynamics
\[
dX_t = K^P(\theta^P - X_t)\,dt + \Sigma\,dW_t^P
\]

This corresponds to the code’s discrete monthly transition equation:

\[
X_t = c + \Phi X_{t-1} + \eta_t
\]

### Paper’s estimation framework
The paper estimates the model in a state-space setting using maximum likelihood. The code does the same using the Kalman filter.

---

## 7. Practical interpretation of the three factors

### Level
Moves the entire yield curve up or down.

### Slope
Makes the curve steeper or flatter, especially at the short end.

### Curvature
Creates or removes the hump in intermediate maturities.

In an economic setting:
- level often tracks the long-run policy and inflation environment,
- slope often reflects expected tightening or easing,
- curvature captures medium-term transition dynamics.

---

## 8. What makes this model useful

This implementation is useful because it combines:

- the **interpretability** of Nelson–Siegel,
- the **statistical tractability** of a Gaussian state-space model,
- and the **theoretical discipline** of no-arbitrage pricing.

That combination is what makes AFNS a practical workhorse model for yield curve analysis.

---

## 9. Summary

This code estimates an AFNS model by:

1. loading and preparing yield data,
2. representing the yield curve with three latent factors,
3. adding the no-arbitrage yield adjustment,
4. estimating the parameters by maximum likelihood with a Kalman filter,
5. smoothing the hidden factors,
6. and decomposing yields into expectations and term premia.

The model is a discrete-time implementation of the AFNS idea developed by Christensen, Diebold, and Rudebusch (2011).
