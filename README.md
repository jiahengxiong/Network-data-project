# Project #1-2-3 – EDFA Profile Regression

## Background

- **Optical amplifiers** (Erbium-Doped Fiber Amplifier, EDFA) compensate power attenuation in optical fibers and guarantee sufficient power at the receiver.
- **Input power profile**: \( P_{in}(\Lambda) = \{ P_{in}(\lambda_1), P_{in}(\lambda_2), \ldots, P_{in}(\lambda_N) \} \)
- **Output power profile**: \( P_{out}(\Lambda) = \{ P_{out}(\lambda_1), P_{out}(\lambda_2), \ldots, P_{out}(\lambda_N) \} \)

### Key Points
- **Gain is not the same across the wavelengths**:
  - It is **linearly tilted** to compensate for different fiber attenuation at different wavelengths.
  - It has additional **ripple** due to imperfections of the production process.
- **Complex transfer function**: \( P_{out}(\Lambda) = f(P_{in}(\Lambda)) \)

### Objective
- Estimate \( P_{out} \) from \( P_{in} \) along the signal path before launching the signal to choose the best wavelength for transmission:
  - To have the flat power profile at the receiver.
  - To have the highest SNR at the receiver.
- **Challenge**: No known analytical model for \( f \). Can we use a ML-model instead?

---

## Dataset 1

- **Source**: TUD, December 2020 [[1]](https://data.dtu.dk/articles/dataset/Input-output_power_spectral_densities_for_three_C-band_EDFAs_and_four_multispan_inline_EDFAd_fiber_optic_systems_of_different_lengths/13135754) [[2]](https://ieeexplore.ieee.org/document/9333297)
- **Input/output power profiles for a single EDFA**:
  - Different gain settings:
    - Total input power to the EDFA varies in the [-9; 9] dBm range.
    - Total output power is 15 dBm.
  - One power measurement in each of 84 channels.
  - 16497 entries with different power profiles.
  - **Synthetic input power profiles** – not real on/off channels.

### Dataset Structure
| Profile Id | Total Power In | Total Power Out | Input Profile (P. Ch. 1, ..., P. Ch. N) | Output Profile (P. Ch. 1, ..., P. Ch. N) |
|------------|----------------|-----------------|----------------------------------------|-----------------------------------------|

---

## Project #1 – EDFA Profile MIMO Regression

### Assignment

- **Brief Summary**:
  - Given input power profile, predict output power profile: regression with multiple inputs and multiple outputs.
  - Use **Deep Neural Network regression**.
  - Use **Dataset 1**.

- **Key Questions**:
  - How many samples of the input profile do we need to characterize EDFA behavior?
    - Sample input power profiles every 1st/2nd/5th/10th/... channel.
  - Train regressors with different input dimensions to predict **full output power profile**.

- **Objective**: Compare the accuracy of output power profile prediction and model complexity.

### Model Diagram
\[
P_{in}(\lambda_1), P_{in}(\lambda_2), P_{in}(\lambda_3), \ldots, P_{in}(\lambda_N)
                \downarrow
               \text{DNN}
                \downarrow
P_{out}(\lambda_1), P_{out}(\lambda_2), P_{out}(\lambda_3), \ldots, P_{out}(\lambda_N)
\]

## References

1. [Input-output power spectral densities for three C-band EDFAs and four multispan inline EDFAd fiber optic systems of different lengths](https://data.dtu.dk/articles/dataset/Input-output_power_spectral_densities_for_three_C-band_EDFAs_and_four_multispan_inline_EDFAd_fiber_optic_systems_of_different_lengths/13135754)
2. [IEEE Explore Document 9333297](https://ieeexplore.ieee.org/document/9333297)
