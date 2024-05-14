# 📋 Wireless Sensing Background

## What is Wireless Sensing

Various sensors and sensor networks have thoroughly extended human perception of the physical world. Nowadays, numerous sensors have been deployed to complete various sensing tasks, resulting in a significant deployment and maintenance overhead. This problem becomes increasingly troublesome when a large sensing scale is in demand. Taking indoor person tracking as an example, a specialized tracking system only covers a room-level area, which is too small compared with the moving region during a person's daily life. Multiple tracking systems are needed to achieve practical sensing coverage in realistic living environments, e.g., houses, campuses, markets, airports, and offices, and the cost inevitably ramps up.

Given the cost limitations of the sensors, many pioneers tried to figure out an alternative solution during the past decade. Nowadays, various types of signals (e.g., Wi-Fi, LoRa, LTE) are filling out our living and working spaces for wireless communication, which can be leveraged to capture the environmental changes without causing extra overhead.

According to the electromagnetism theory, the radio signals emitted by the transmitter (Tx) experience various physical phenomena such as reflection, diffraction, and scattering during the propagation process and form into multiple propagation paths.

In this way, the superimposed multipath signals collected by the receiver carry spatial information about the signal propagation environment. Relying only on the ambient wireless signals and ubiquitous communication devices, wireless sensing emerges as a novel paradigm for environment sensing.

In recent years, wireless sensing technology has attracted many research interests to bring wireless sensing from the imagination into reality, by boosting sensing granularity, improving system robustness, and exploring application scenarios. Many of their works on wireless sensing have been published in flagship conferences and journals, such as ACM SIGCOMM, ACM MobiCom, ACM MobiSys, IEEE INFOCOM, USENIX NSDI, IEEE/ACM ToN, IEEE JSAC, and IEEE TMC. In addition, many famous companies are also exploring the productization of sensorless sensing, launching various IoT devices for human-computer interaction, security monitoring, and health care.

## Comparison of Wireless Sensing and Computer Vision

&#x20;\*\* Fig. 1. Comparision of vision-based sensing and RF-based sensing processes. \*\*

Typical RF signals (300 kHz - 300 GHz) and visible light signals (380 THz - 750 THz) are essentially electromagnetic (EM) waves. When propagating in our physical world, the EM waves experience a variety of physical phenomena such as reflection, diffraction, and scattering. Multipath signals are eventually superimposed and received by the receiver. Therefore, the received superimposed signals carry the physical information of the signal propagation space.

Both the RF-based and the vision-based sensing algorithms share similar processes. They first analyze the received signals (radio signal at the antenna or visible light at the camera lens), from which the features reflecting the propagation space are extracted and finally resolved by algorithms to realize the sensing of the surrounding environment.

Compared with vision-based sensing, wireless sensing solutions have unique advantages such as high coverage, pervasiveness, low cost, and robustness under adverse light and texture scenarios .

## Wireless Sensing Applications

Wireless sensing systems are capable of perceiving changes in surrounding environments, objects, and human bodies. In this subsection, we take _passive human sensing_ applications as an example, which refers to a human-centered sensing application that doesn't require the user to carry any device. Therefore, such an sensing application is also termed as _device-free sensing_ or _non-invasive sensing_. Passive human sensing enables a wide range of applications, including smart homes, security surveillance, and health care.

In **smart home** applications, passive human sensing recognizes a person's behavior or intention based on the user's physical locations, gestures, and postures. Passive human sensing brings a better user experience without imposing restrictions on the user. For example, users can remotely control electrical devices, e.g., television, computer, or washer, by merely performing gestures in the air. Likewise, when playing video games, users can interact with the computer by performing different postures .

In **security surveillance** applications, traditional methods adopt infrared or RGB cameras to monitor illegal invasions, protect valuable properties, and deal with emergencies. However, cameras are constrained by the limited field of view or blockage of opaque or metallic objects, rendering these methods to fail when the target is not in the Line-of-Sight (LoS) area of the surveillance camera or hidden behind other objects. In contrast to visual surveillance, wireless sensing technology leverages radio signals, which provide omnidirectional coverage around the wireless devices and are less prone to blockages. For example, wireless signals can be used to detect illegal intrusions. Besides, they can also be used to detect if properties have been moved from their original places.

In **health care** applications, passive human sensing can be leveraged to detect vital signals such as human respiration, heartbeat, gait, and accidental fall. Specifically, some researchers have exploited Wi-Fi signals to detect human respiration for sleep monitoring. Some other workshave extracted gait patterns from Wi-Fi signals to recognize human identity. Recently, Wi-Fi signals have been further used to detect accidental falls to relieve the need for wearable sensors.
