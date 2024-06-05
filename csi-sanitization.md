# 🌊 CSI Sanitization

{% hint style="info" %}
All the code and data in this tutorial are available. Click [**here**](https://github.com/Guoxuan-Chi/Wireless-Sensing-Tutorial/releases/download/v0.1.0-alpha/Hands-on-Wireless-Sensing.zip) to download!&#x20;
{% endhint %}

{% hint style="warning" %}
This section is an **advanced content**, and is written for developers who already have enough experience in designing wireless localization and tracking systems. If you are a beginner, or a researcher only focusing on deep-learning-based wireless sensing, you may skip this section.&#x20;
{% endhint %}

The wireless sensing models and features described in previous sections are consistent with the EM propagation theory and geometry. However, they don't consider the various types of noise caused by imperfect implementations of transceiver hardware. This section focuses on various CSI error sources and the corresponding error cancellation algorithms.

For ease of illustration, code implementation for sanitization is provided, which is a `main` function to call different error cancellation functions and test their performance.

```matlab
%{
  CSI Sanitization Algorithm for Wi-Fi sensing.  
  - Input: csi data used for calibration, and csi data that need to be sanitized.
  - Output: sanitized csi data.

  To use this script, you need to:
  1. Make sure the csi data have been saved as .mat files.
  2. Check the .mat file to make sure they are in the correct form. 
  3. Set the parameters.
  
  Note that in this algorithm, the csi data should be a 4-D tensor with the size of [T S A L]:
  - T indicates the number of csi packets;
  - S indicates the number of subcarriers;
  - A indicates the number of antennas (i.e., the STS number in a MIMO system);
  - L indicates the number of HT-LTFs in a single PPDU;
  Say if we collect a 10 seconds csi data steam at 1 kHz sample rate (T = 10 * 1000), from a 3-antenna AP (A = 3),  with 802.11n standard (S = 57 subcarrier), without only one spatial stream (L = 1), the data size should be [10000 57 3 1].
%}

clear all;
addpath(genpath(pwd));

%% 0. Set parameters.
% Path of the calibration data;
calib_file_name = './data/csi_calib_test.mat';
% Path for storing the generated calibration templated.
calib_template_name = './data/calib_template_test.mat';
% Path of the raw CSI data.
src_file_name = './data/csi_src_test.mat';
% Path for storing the sanitized CSI data.
dst_file_name = './data/csi_dst_test.mat';

% Speed of light.
global c;
c = physconst('LightSpeed');
% Bandwidth.
global bw;
bw = 20e6;
% Subcarrier frequency.
global subcarrier_freq;
subcarrier_freq = linspace(5.8153e9, 5.8347e9, 57);
% Subcarrier wavelength.
global subcarrier_lambda;
subcarrier_lambda = c ./ subcarrier_freq;

% Antenna arrangement.
antenna_loc = [0, 0, 0; 0.0514665, 0, 0; 0, 0.0514665, 0]';
% Set the linear range of the CSI phase, which varies with NIC types.
linear_interval = (20:38)';

%% 1. Read the csi data for calibration and sanitization.
% Load the calibration data. 
csi_calib = load(calib_file_name).csi; % CSI for calibration.
% Load the raw CSI data.
csi_src = load(src_file_name).csi;      % Raw CSI.

%% 2. Choose different functions according to your task.
% Use cases:
% Make calibration template.
csi_calib_template = set_template(csi_calib, linear_interval, calib_template_name);
% Directly load the generated template.
csi_calib_template = load(calib_template_name).csi;
% Remove the nonlinear error.
csi_remove_nonlinear = nonlinear_calib(csi_src, csi_calib_template);
% Remove the STO (a.k.a SFO and PBD) by conjugate mulitplication.
csi_remove_sto = sto_calib_mul(csi_src);
% Remove the STO (a.k.a SFO and PBD) by conjugate division.
csi_remove_sto = sto_calib_div(csi_src);
% Estimate the CFO by frequency tracking.
est_cfo = cfo_calib(csi_src);
% Estimate the RCO.
est_rco = rco_calib(csi_calib);

%% 3. Save the sanitized data as needed.
csi = csi_remove_sto;
save(dst_file_name, 'csi');

%% 4. Perform various wireless sensing tasks.
% Test example 1: angle/direction estimation with imperfect CSI.
[packet_num, subcarrier_num, antenna_num, ~] = size(csi_src);
est_rco = rco_calib(csi_calib);
zero_rco = zeros(antenna_num, 1);
aoa_mat_error = naive_aoa(csi_src, antenna_loc, zero_rco);
aoa_mat = naive_aoa(csi_src, antenna_loc, est_rco);
aoa_gt = [0; 0; 1];
error_1 = mean(acos(aoa_gt' * aoa_mat_error));
error_2 = mean(acos(aoa_gt' * aoa_mat));
disp("Angle estimation error (in deg) without RCO removal: " + num2str(error_1));
disp("Angle estimation error (in deg) with RCO removal: " + num2str(error_2));

% Test example 2: intrusion detection with CSI.
csi_sto_calib = sto_calib_div(csi_src);
intrusion_flag_raw = naive_intrusion(csi_src, 3);
intrusion_flag_sanitized = naive_intrusion(csi_sto_calib, 3);
disp("Intrusion detection result without SFO/PDD removal: " + num2str(intrusion_flag_raw));
disp("Intrusion detection result with SFO/PDD removal: " + num2str(intrusion_flag_sanitized));
```

## Nonlinear Amplitude and Phase

The nonlinear amplitude and phase errors are caused by the imperfect analog domain filter implementation inside the hardware. Specifically, it causes the extracted CSI amplitude and phase to be equivalently processed by a nonlinear function. Let $$f(\cdot)$$ and $$g(\cdot)$$ be the nonlinear modes of CSI amplitude and phase, respectively, the errorous CSI can be written as:

$$
\tilde{H}(i,j,k) = \sum_{n = 1}^{N} {f(\alpha_{n}) e^{-jg(\phi_{n}(i,j,k))}} + N(i,j,k). (29)
$$

Specifically, during the OFDM modulation process, each subcarrier of should have the same gain. In other words, the amplitude-frequency characteristic of CSI should be a horizontal straight line when a coaxial cable is used to connect the transceiver ports. However, actual measurements show that even without the multipath radio channel, there is still a similar "frequency selective fading" characteristic, \ie, the gain of each frequency band is different, showing an M-shaped amplitude-frequency characteristic curve. Similarly, when using a coaxial cable to connect the transceiver port, the CSI phase-frequency characteristics obtained by the NIC are not an ideal straight line with slope, but an S-shaped curve with certain nonlinearity.

After extensive research and experiments, we have observed two facts.

* First, for a specific type of NIC, the nonlinear amplitude/phase error of CSI is fixed. This means that the correction task can be accomplished if we use a known length coaxial cable connected to the transceiver port. Before performing the sensing task, measure a representative set of CSI amplitude and phase, record the nonlinear characteristics, and eliminate the nonlinearity in the subsequent steps.
* Second, we observe that the middle part of the subcarrier is free of nonlinear errors, and the nonlinear characteristics of both sides are also fixed.&#x20;

Therefore, the function below the code performs the following steps to tackle the CSI nonlinearity:

* Read in a set of CSI data measured using a coaxial cable.
* Get its amplitude and phase.
* Normalize its amplitude and record it as the amplitude template.
* Unwrap the phases, then perform a linear fit to the middle part of its subcarriers. Subtract the linear fit result to obtain the nonlinear phase error template.

Finally, the nonlinear amplitude and nonlinear phase components are saved in the form of $$\alpha e^{j\phi}$$, which indicates the nonlinear error template of a specific type of NIC.

```c
function [csi_calib_template] = set_template(csi_calib, linear_interval, calib_template_name)
    % set_template
    % Input:
    %   - csi_calib is the reference csi at given distance and angle; [T S A L]
    %   - linear_interval is the linear range of the csi phase, which varies across different types of NICs;
    %   - calib_template_name is the saved path of the generated template;
    % Output:
    %   - csi_calib_template is the generated template for csi calibration; [1 S A L]

    [packet_num, subcarrier_num, antenna_num, extra_num] = size(csi_calib);
    csi_amp = abs(csi_calib);                       % [T S A L]
    csi_phase = unwrap(angle(csi_calib), [], 2);    % [T S A L]
    csi_amp_template = mean(csi_amp ./ mean(csi_amp, 2), 1); % [1 S A L]
    nonlinear_phase_error = zeros(size(csi_calib));          % [T S A L]
    for p = 1:packet_num
        for a = 1:antenna_num
            for e = 1:extra_num
                linear_model = fit(linear_interval, squeeze(csi_phase(p, linear_interval, a, e))', 'poly1');
                nonlinear_phase_error(p, :, a, e) = csi_phase(p, :, a, e) - linear_model(1:subcarrier_num)';
            end
        end
    end
    csi_phase_template = mean(nonlinear_phase_error, 1); % [1 S A L]
    csi_phase_template(1, linear_interval, :, :) = 0;
    csi_calib_template = csi_amp_template .* exp(1i * csi_phase_template); % [1 S A L]
    csi = csi_calib_template;
    save(calib_template_name, 'csi'); % [1 S A L]
end
```

After getting the nonlinear error template, the sanitization process begins. For the raw CSI data `csi_data` collected in real time, we divide it by the error template `csi_calib`. This operation is equivalent to "dividing the original amplitude by the normalized nonlinear amplitude" and "subtracting the original phase from the nonlinear phase", so that the returned CSI data `csi_proc` has sanitized amplitude and phase.

```c
function [csi_remove_nonlinear] = nonlinear_calib(csi_calib, csi_calib_template)
    % nonlinear_calib
    % Input:
    %   - csi_src is the raw csi which needs to be calibrated; [T S A L]
    %   - csi_calib_template is the reference csi for calibration; [1 S A L]
    % Output:
    %   - csi_remove_nonlinear is the csi data in which the nonlinear error has been eliminated; [T S A L]

    csi_amp = abs(csi_calib);                       % [T S A L]
    csi_phase = unwrap(angle(csi_calib), [], 2);    % [T S A L]
    csi_unwrap = csi_amp .* exp(1i * csi_phase);     % [T S A L]
    % Broadcasting is performed.
    csi_remove_nonlinear = csi_unwrap ./ csi_calib_template;
end
```

## Automatic Gain Control Uncertainty

Automatic gain control (AGC) induces a random gain $$\beta$$ in each received CSI packet.

$$
\tilde{H}(i,j,k) = \sum_{n = 1}^{N} {\beta_{i} \alpha_{n} e^{-j\phi_{n}(i,j,k)}} + N(i,j,k). (30)
$$

There are two ways to eliminate the AGC error: 1) Disable the AGC function of the wireless driver; 2) Compensate the amplitude of the measured CSI based on the reported AGC.

```c
function [csi_remove_agc] = agc_calib(csi_src, csi_agc)
    % rco_calib
    % Input:
    %   - csi_src is the raw csi which needs to be calibrated; [T S A L]
    %   - csi_agc is the AGC amplitude reported by the NIC; [T, 1]
    % Output:
    %   - csi_remove_agc is the csi data in which the AGC uncertainty has been eliminated; [T S A L]

    % Broadcasting is performed.
    csi_remove_agc = csi_src ./ csi_agc;
end
```

## Radio Chain Offset

Radio chain offset (RCO) is the random phase variation $$\epsilon_{\phi}$$ introduced between different Tx/Rx chains (transceiver antenna pairs). The RCO is reset each time the NIC is powered up.

$$
\tilde{\phi}_{n}(i,j,k) = 2\pi(f_c + \Delta f_j + f_{D}) \tau_n(i, j, k) + \epsilon_{\phi}, (31)
$$

RCO induces a biast of the phase-frequency characteristic curve. It undermines the accuracy of features such as AoA or ToF. Fortunately, we found that this type of phase deviation is consistent between each successive packet sent and therefore doesn't affect the performance of temporal tracking or sensing, and that this type of error can be eliminated by the following steps:

* Once the NIC power-up, connect the transceiver port using a coaxial cable of known length and record the phase information `calib_phase`.
* During subsequent measurements, subtracting this phase information from the measured phase `csi_phase`.

```c
function [est_rco] = rco_calib(csi_calib)
    % rco_calib
    % Input:
    %   - csi_calib is the reference csi at given distance and angle; [T S A L]
    % Output:
    %   - est_rco is the estimated RCO; [A 1]

    antenna_num = size(csi_calib, 3);
    csi_phase = unwrap(angle(csi_calib), [], 1);    % [T S A L]
    avg_phase = zeros(antenna_num, 1);
    for a = 1:antenna_num
        avg_phase(a, 1) = mean(csi_phase(:, :, a, 1), 'all');
    end
    est_rco = avg_phase - avg_phase(1, 1);
end
```

## Central Frequency Offset

Central frequency offset (CFO), which is caused by the frequency desynchronization on both sides of the transceiver, leads to random frequency shift $$\epsilon_{f}$$ of each received CSI.

$$
\tilde{\phi}_{n}(i,j,k) = 2\pi(f_c + \Delta f_j + f_{D} + \epsilon_{f}) \tau_n(i, j, k) = \phi_{n}(i,j,k) + \epsilon_{f}\tau_n(i, j, k). (32)
$$

The CFO induces an extra bias (i.e., an overall up and down shift) of the phase-frequency characteristic curve.

To eliminate CFO, we need to insert multiple HT-LTFs in the same PPDU (Wi-Fi data frame), and therefore obtaining multiple CSI measurements. Since the time interval between multiple HT-LTFs is strictly controlled to $$4 \mu s$$ according to the 802.11 protocol, the phase difference between two HT-LTFs is induced by the CFO within $$\Delta t = 4\mu s$$. Thus, the approximate value of the CFO can be recovered.

```c
function [est_cfo] = cfo_calib(csi_src)
    % cfo_calib
    % Input:
    %   - csi_src is the csi data with two HT-LTFs; [T S A L]
    % Output:
    %   - est_cfo is the estimated frequency offset; 

    delta_time = 4e-6;
    phase_1 = angle(csi_src(:, :, :, 1));
    phase_2 = angle(csi_src(:, :, :, 2));
    phase_diff = mean(phase_2 - phase_1, 3); % [T S A 1]
    est_cfo = mean(phase_diff ./ delta_time, 2);
end
```

## Sampling Frequency Offset and Packet Detection Delay

The sampling frequency offset (SFO), which appears to be an error in frequency domain, are generally considered as an equivalent time shift due to frequency asynchrony. The packet detection delay (PDD) is a time delay. Therefore, despite their distinct causes, they are often discussed together as a "time offset" together.

$$
\tilde{\phi}_{n}(i,j,k) = 2\pi(f_c + \Delta f_j + f_{D}) (\tau_n(i, j, k) + \epsilon_{t}) = \phi_{n}(i,j,k) + 2\pi(f_c + \Delta f_j + f_{D}) \epsilon_{t}. (33)
$$

This type of error is critical because the time delay can be confused with real ToF and thus affect the accuracy of the ranging accuarcy. Specifically, this deviation will be characterized in the phase-frequency characteristic curve as a change in slope, since this time deviation causes different phase changes with different sub-bands $$\Delta f_j$$.

Currently, there is no "perfect algorithm" to solve this type of error. Conjugate multiplication and division are the only two methods to eliminate the SFO and PDD. The code of both of them are listed below. By appling the conjugate multiplication or division, the $$\epsilon_{t}$$ is eliminated, at the cost of losing absolute ToF measurement.

```c
function [csi_remove_sto] = sto_calib_mul(csi_src)
    % sto_calib_mul
    % Input:
    %   - csi_src is the csi data with sto; [T S A L]
    % Output:
    %   - csi_remove_sto is the csi data without sto; [T S A L]

    antenna_num = size(csi_src, 3);
    csi_remove_sto = zeros(size(csi_src));
    for a = 1:antenna_num
        a_nxt = mod(a, antenna_num) + 1;
        csi_remove_sto(:, :, a, :) = csi_src(:, :, a, :) .* conj(csi_src(:, :, a_nxt, :));
    end
end
```

```c
function [csi_remove_sto] = sto_calib_div(csi_src)
    % sto_calib_div
    % Input:
    %   - csi_src is the csi data with sto; [T S A L]
    % Output:
    %   - csi_remove_sto is the csi data without sto; [T S A L]

    antenna_num = size(csi_src, 3);
    csi_remove_sto = zeros(size(csi_src));
    for a = 1:antenna_num
        a_nxt = mod(a, antenna_num) + 1;
        csi_remove_sto(:, :, a, :) = csi_src(:, :, a, :) ./ csi_src(:, :, a_nxt, :);
    end
end
```

To sum up, there are various forms of errors in Wi-Fi CSI measurements, including fixed bias and random errors. Each of them have different impacts on the localization, tracking, and sensing tasks. The erroneous CSI form can be finally written as:

$$
\tilde{H}(i,j,k) = \sum_{n = 1}^{N} {\beta_{i}f(\alpha_{n}) e^{-jg(\phi_{n}(i,j,k))}} + N(i,j,k), \\ \tilde{\phi}_{n}(i,j,k) = 2\pi(f_c + \Delta f_j + f_{D} + \epsilon_{f}) (\tau_n(i, j, k) + \epsilon_{t}) + \epsilon_{\phi}. (34)
$$
