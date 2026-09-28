---
layout: page
title: Research
permalink: /research/
nav: true
nav_order: 2
---

My research lies at the intersection of control, machine learning, and formal verification, with the goal of enabling safe and reliable autonomy in complex, sometimes multi-agent environments. As complex autonomous systems rely on learned models and decision-making algorithms, it is important to have good performance, understand the behavior of these autonomous systems, and provide quantitative guarantees on their safety, whether deterministic or probabilistic. I am particularly interested in how we can combine learning, statistical tools, and control-theoretic tools to reason about uncertainty, interactions between agents, and the behavior of learning-enabled autonomous systems.


Learning-based policies can enable high-dimensional autonomous systems to operate in complex and uncertain environments; however, their behavior can be difficult to characterize with conventional model-based verification techniques. **My research addresses this via a focus on data-driven and probabilistic methods for verifying learning-enabled dynamical systems**, with a particular focus on reachability analysis.

I am interested in obtaining rigorous guarantees from data while keeping verification computationally and statistically tractable. In particular, I investigate methods for estimating **reachable (reach-avoid) sets with probabilistic guarantees** and for reducing the computational and sample requirements needed to certify large regions of the state space. As opposed to relying solely on empirical evaluation, these methods provided quantitative guarantees that can be used to reason about the behavior of learned controllers under uncertainty. A key focus of my recent work is developing **hierarchical approaches to probabilistic verification** that combines global and local information. By identifying regions of interest and constructing tighter local reachability estimates, my works seeks to make verification more scalable while retaining rigorous guarantees. I am particularly interested in applying these methods to **multi-agent systems**, where interactions between agents introduce additional uncertainty and where safety decisions must often be made online.

### Learning-Based Control
I am also interested in understanding what modern machine learning architectures can learn about **control and dynamical systems**. One direction of my work investigates whether transformer models can perform low-level control through **in-context learning**, using information provided during deployment to adapt their behavior across different system instances.

I study transformer-based control on nonlinear systems, including unstable systems such as the Cartpole and Acrobot. In particular, I am interested in how performance changes across different system parameters, how much context is required for effective control, and how these models behave **in-distribution and out-of-distribution**. More broadly, this work seeks to understand the capabilities and limitations of modern sequence models when they are used to control dynamical systems.

### Safe Control and Multi-Agent Systems
A broader goal of my research is to develop methods for **safe decision making in multi-agent environments**, where an autonomous agent must act while accounting for other agent whose behavior may be uncertain, adaptive, or only partially observable. In many existing approaches to safe control and verification, the ego agent is assumed to have access to the intent or policy of other agents. I am interested in relaxing this assumption by incorporating **intent estimation and uncertainty about other agents' behavior directly into control and verification**.

This raises fundamental questions about how an autonomous system should reason about another agent while simultaneously making safety-critical decisions: how can intent be inferred from observations, how should uncertainty in that estimate be represented, and how can those estimates be incorporated into **reachability analysis and safe control**? My ongoing work explores this intersection of **intent-aware autonomy, uncertainty quantification, and formal safety guarantees**. Ultimately, I aim to develop learning-enabled autonomous systems that can act affectively in complex environments while simultaneously reasoning about uncertainty and safety while interacting with other agents.

