# Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking

Kalman filter (KF) based methods for multi-object tracking (MOT) make an assumption that objects move linearly. While this assumption is acceptable for very short periods of occlusion, linear estimates of motion for prolonged time can be highly inaccurate. Moreover, when there is no measurement available to update Kalman filter parameters, the standard convention is to trust the priori state estimations for posteriori update. This leads to the accumulation of errors during a period of occlusion. The error causes significant motion direction variance in practice. In this work, we show that a basic Kalman filter can still obtain state-of-the-art tracking performance if proper care is taken to fix the noise accumulated during occlusion. Instead of relying only on the linear state estimate (i.e., estimation-centric approach), we use object observations (i.e., the measurements by object detector) to compute a virtual trajectory over the occlusion period to fix the error accumulation of filter parameters during the occlusion period. This allows more time steps to correct errors accumulated during occlusion. We name our method Observation-Centric SORT (OC-SORT). It remains Simple, Online, and Real-Time but improves robustness during occlusion and non-linear motion. Given off-the-shelf detections as input, OC-SORT runs at 700+ FPS on a single CPU. It achieves state-of-the-art on multiple datasets, including MOT17, MOT20, KITTI, head tracking, and especially DanceTrack where the object motion is highly non-linear. The code and models are available at https://github.com/noahcao/OC_SORT.

## 1 Introduction

We aim to develop a motion model-based multi-object tracking (MOT) method that is robust to occlusion and non-linear motion. Most existing motion model-based algorithms assume that the tracking targets have a constant velocity within a time interval, which is called the linear motion assumption. This assumption breaks in many practical scenarios, but it still works because when the time interval is small enough, the object’s motion can be reasonably approximated as linear. In this work, we are motivated by the fact that most of the errors from motion model-based tracking methods occur when occlusion and non-linear motion happen together. To mitigate the adverse effects caused, we first rethink current motion models and recognize some limitations. Then, we propose addressing them for more robust tracking performance, especially in occlusion.

As the main branch of motion model-based tracking, filtering-based methods assume a transition function to predict the state of objects on future time steps, which are called state “estimations”. Besides estimations, they leverage an observation model, such as an object detector, to derive the state measurements of target objects, also called “observations”. Observations usually serve as auxiliary information to help update the posteriori parameters of the filter. The trajectories are still extended by the state estimations. Among this line of work, the most widely used one is SORT [3], which uses a Kalman filter (KF) to estimate object states and a linear motion function as the transition function between time steps. However, SORT shows insufficient tracking robustness when the object motion is non-linear, and no observations are available when updating the filter posteriori parameters.

In this work, we recognize three limitations of SORT. First, although the high frame rate is the key to approximating the object motion as linear, it also amplifies the model’s sensitivity to the noise of state estimations. Specifically, between consecutive frames of a high frame-rate video, we demonstrate that the noise of displacement of the object can be of the same magnitude as the actual object displacement, leading to the estimated object velocity by KF suffering from a significant variance. Also, the noise in the velocity estimate will accumulate into the position estimate by the transition process. Second, the noise of state estimations by KF is accumulated along the time when there is no observation available in the update stage of KF. We show that the error accumulates very fast with respect to the time of the target object’s being untracked. The noise’s influence on the velocity direction often makes the track lost again even after re-association. Last, given the development of modern detectors, the object state by detections usually has lower variance than the state estimations propagated along time steps by a fixed transition function in filters. However, SORT is designed to prolong the object trajectories by state estimations instead of observations.

To relieve the negative effect of these limitations, we propose two main innovations in this work. First, we design a module to use object state observations to reduce the accumulated error during the track’s being lost in a backcheck fashion. To be precise, besides the traditional stages of predict and update, we add a stage of re-update to correct the accumulated error. The re-update is triggered when a track is re-activated by associating to an observation after a period of being untracked. The re-update uses virtual observations on the historical time steps to prevent error accumulation. The virtual observations come from a trajectory generated using the last-seen observation before untracked and the latest observation re-activating this track as anchors. We name it Observation-centric Re-Update (ORU).

Besides ORU, the assumption of linear motion provides the consistency of the object motion direction. But this cue is hard to be used in SORT’s association because of the heavy noise in direction estimation. But we propose an observation-centric manner to incorporate the direction consistency of tracks in the cost matrix for the association. We name it Observation-Centric Momentum (OCM). We also provide analytical justification for the noise of velocity direction estimation in practice.

The proposed method, named as Observation-Centric SORT or OC-SORT in short, remains simple, online, real-time and significantly improves robustness over occlusion and non-linear motion. Our contributions are summarized as the following:

- 1. We recognize, analytically and empirically, three limitations of SORT, i.e. sensitivity to the noise of state estimations, error accumulation over time, and being estimation-centric;
We recognize, analytically and empirically, three limitations of SORT, i.e. sensitivity to the noise of state estimations, error accumulation over time, and being estimation-centric;

- 2. We propose OC-SORT for tracking under occlusion and non-linear motion by fixing SORT’s limitations. It achieves state-of-the-art performance on multiple datasets in an online and real-time fashion.
We propose OC-SORT for tracking under occlusion and non-linear motion by fixing SORT’s limitations. It achieves state-of-the-art performance on multiple datasets in an online and real-time fashion.

## 2 Related Works

Many modern MOT algorithms [3, 11, 63, 73, 70] use motion models. Typically, these motion models use Bayesian estimation [34] to predict the next state by maximizing a posterior estimation. As one of the most classic motion models, Kalman filter (KF) [30] is a recursive Bayes filter that follows a typical predict-update cycle. The true state is assumed to be an unobserved Markov process, and the measurements are observations from a hidden Markov model [44]. Given that the linear motion assumption limits KF, follow-up works like Extended KF [52] and Unscented KF [28] were proposed to handle non-linear motion with first-order and third-order Taylor approximation. However, they still rely on approximating the Gaussian prior assumed by KF and require motion pattern assumption. On the other hand, particle filters [22] solve the non-linear motion by sampling-based posterior estimation but require exponential order of computation. Therefore, these variants of Kalman filter and particle filters are rarely adopted in the visual multi-object tracking and the mostly adopted motion model is still based on Kalman filter [3].

As a classic computer vision task, visual multi-object tracking is traditionally approached from probabilistic perspectives, e.g. joint probabilistic association [1]. And modern video object tracking is usually built upon modern object detectors [46, 48, 74]. SORT [3] adopts the Kalman filter for motion-based multi-object tracking given observations from deep detectors. DeepSORT [63] further introduces deep visual features [51, 23] into object association under the framework of SORT. Re-identification-based object association[63, 42, 71] has also become popular since then but falls short when scenes are crowded and objects are represented coarsely (e.g. enclosed by bounding boxes), or object appearance is not distinguishable. More recently, transformers [58] have been introduced to MOT [39, 69, 55, 8] to learn deep representations from both visual information and object trajectories. However, their performance still has a significant gap between state-of-the-art tracking-by-detection methods in terms of both accuracy and time efficiency.

## 3 Rethink the Limitations of SORT

In this section, we review Kalman filter and SORT [3]. We recognize some of their limitations, which are significant with occlusion and non-linear object motion. We are motivated to improve tracking robustness by fixing them.

### 3.1 Preliminaries

Kalman filter (KF) [30] is a linear estimator for dynamical systems discretized in the time domain. KF only requires the state estimations on the previous time step and the current measurement to estimate the target state on the next time step. The filter maintains two variables, the posteriori state estimate 𝐱\mathbf{x}, and the posteriori estimate covariance matrix 𝐏\mathbf{P}. In the task of object tracking, we describe the KF process with the state transition model 𝐅\mathbf{F}, the observation model 𝐇\mathbf{H}, the process noise 𝐐\mathbf{Q}, and the observation noise 𝐑\mathbf{R}. At each step tt, given observations 𝐳t\mathbf{z}_{t}, KF works in an alternation of predict and update stages:

|  |  | predict{𝐱^t|t−1=𝐅t​𝐱^t−1|t−1𝐏t|t−1=𝐅t​𝐏t−1|t−1​𝐅t⊤+𝐐t,\displaystyle\text{\text{{predict}}}\left\{\begin{aligned} &\hat{\mathbf{x}}_{t|t-1}=\mathbf{F}_{t}\hat{\mathbf{x}}_{t-1|t-1}\\ &\mathbf{P}_{t|t-1}=\mathbf{F}_{t}\mathbf{P}_{t-1|t-1}\mathbf{F}_{t}^{\top}+\mathbf{Q}_{t}\\ \end{aligned},\right. |  | (1) |
|---|---|---|---|---|
|  |  | update{𝐊t=𝐏t|t−1​𝐇t⊤​(𝐇t​𝐏t|t−1​𝐇t⊤+𝐑t)−1𝐱^t|t=𝐱^t|t−1+𝐊t​(𝐳t−𝐇t​𝐱^t|t−1)𝐏t|t=(𝐈−𝐊t​𝐇t)​𝐏t|t−1.\displaystyle\text{\text{{update}}}\left\{\begin{aligned} \mathbf{K}_{t}&=\mathbf{P}_{t|t-1}\mathbf{H}_{t}^{\top}(\mathbf{H}_{t}\mathbf{P}_{t|t-1}\mathbf{H}_{t}^{\top}+\mathbf{R}_{t})^{-1}\\ \hat{\mathbf{x}}_{t|t}&=\hat{\mathbf{x}}_{t|t-1}+\mathbf{K}_{t}(\mathbf{z}_{t}-\mathbf{H}_{t}\hat{\mathbf{x}}_{t|t-1})\\ \mathbf{P}_{t|t}&=(\mathbf{I}-\mathbf{K}_{t}\mathbf{H}_{t})\mathbf{P}_{t|t-1}\end{aligned}.\right. |  |  |

The stage of predict is to derive the state estimations on the next time step tt. Given a measurement of target states on the next step tt, the stage of update aims to update the posteriori parameters in KF. Because the measurement comes from the observation model 𝐇\mathbf{H}, it is also called “observation” in many scenarios.

SORT [3] is a multi-object tracker built upon KF. The KF’s state 𝐱\mathbf{x} in SORT is defined as 𝐱=[u,v,s,r,u˙,v˙,s˙]⊤\mathbf{x}=[u,v,s,r,\dot{u},\dot{v},\dot{s}]^{\top}, where (uu, vv) is the 2D coordinates of the object center in the image. ss is the bounding box scale (area) and rr is the bounding box aspect ratio. The aspect ratio rr is assumed to be constant. The other three variables, u˙\dot{u}, v˙\dot{v} and s˙\dot{s} are the corresponding time derivatives. The observation is a bounding box 𝐳=[u,v,w,h,c]⊤\mathbf{z}=[u,v,w,h,c]^{\top} with object center position (u,vu,v), object width ww, and height hh and the detection confidence cc respectively. SORT assumes linear motion as the transition model 𝐅\mathbf{F} which leads to the state estimation as

|  | ut+1=ut+u˙t​Δ​t,vt+1=vt+v˙t​Δ​t.u_{t+1}=u_{t}+\dot{u}_{t}\Delta t,\quad v_{t+1}=v_{t}+\dot{v}_{t}\Delta t. |  | (2) |
|---|---|---|---|

To leverage KF (Eq 1) in SORT for visual MOT, the stage of predict corresponds to estimating the object position on the next video frame. And the observations used for the update stage usually come from a detection model. The update stage is to update Kalman filter parameters and does not directly edit the tracking outcomes.

When the time difference between two steps is constant during the transition, e.g., the video frame rate is constant, we can set Δ​t=1\Delta t=1. When the video frame rate is high, SORT works well even when the object motion is non-linear globally, (e.g. dancing, fencing, wrestling) because the motion of the target object can be well approximated as linear within short time intervals. However, in practice, observations are often absent on some time steps, e.g. the target object is occluded in multi-object tracking. In such cases, we cannot update the KF parameters by the update operation as in Eq. 1 anymore. SORT uses the priori estimations directly as posterior. We call this “dummy update”, namely

|  | 𝐱^t|t=𝐱^t|t−1,𝐏t|t=𝐏t|t−1.\hat{\mathbf{x}}_{t|t}=\hat{\mathbf{x}}_{t|t-1},\mathbf{P}_{t|t}=\mathbf{P}_{t|t-1}. |  | (3) |
|---|---|---|---|

The philosophy behind such a design is to trust estimations when no observations are available to supervise them. We thus call the tracking algorithms following this scheme “estimation-centric”. However, we will see that this estimation-centric mechanism can cause trouble when non-linear motion and occlusion happen together.

### 3.2 Limitations of SORT

In this section, we identify three main limitations of SORT which are connected. This analysis lays the foundation of our proposed method.

Now we show that SORT is sensitive to the noise from KF’s state estimations. To begin with, we assume that the estimated object center position follows u∼𝒩⁡(μu,σu2)u\sim\mathcal{N}(\mu_{u},\sigma_{u}^{2}) and v∼𝒩⁡(μv,σv2)v\sim\mathcal{N}(\mu_{v},\sigma_{v}^{2}), where (μu,μv)(\mu_{u},\mu_{v}) is the underlying true position. Then, if we assume that the state noises are independent on different steps, by Eq.2, the object speed between two time steps, t⟶t+Δ​tt\longrightarrow t+\Delta t, is

|  |  | u˙=ut+Δ​t−utΔ​t,v˙=vt+Δ​t−vtΔ​t,\displaystyle\dot{u}=\frac{u_{t+\Delta t}-u_{t}}{\Delta t},\quad\quad\dot{v}=\frac{v_{t+\Delta t}-v_{t}}{\Delta t}, |  | (4) |
|---|---|---|---|---|

making the noise of estimated speed δu˙∼𝒩⁡(0,2​σu2(Δ​t)2)\delta_{\dot{u}}\sim\mathcal{N}(0,\frac{2\sigma_{u}^{2}}{(\Delta t)^{2}}), δv˙∼𝒩⁡(0,2​σv2(Δ​t)2)\delta_{\dot{v}}\sim\mathcal{N}(0,\frac{2\sigma_{v}^{2}}{(\Delta t)^{2}}). Therefore, a small Δ​t\Delta t will amplify the noise. This suggests that SORT will suffer from the heavy noise of velocity estimation on high-frame-rate videos. The analysis above is simplified from the reality. In pratice, velocity won’t be determined by the state on future time steps. For a more strict analysis, please refer to Appendix G.

Moreover, for most multi-object tracking scenarios, the target object displacement is only a few pixels between consecutive frames. For instance, the average displacement is 1.93 pixels and 0.65 pixels along the image width and height for the MOT17 [41] training dataset. In such a case, even if the estimated position has a shift of only a single pixel, it causes a significant variation in the estimated speed. In general, the variance of the speed estimation can be of the same magnitude as the speed itself or even greater. This will not make a massive impact as the shift is only of few pixels from the ground truth on the next time step and the observations, whose variance is independent of the time, will be able to fix the noise when updating the posteriori parameters. However, we find that such a high sensitivity to state noise introduces significant problems in practice after being amplified by the error accumulation across multiple time steps when no observation is available for KF update.

For analysis above in Eq. 4, we assume the noise of the object state is i.i.d on different time steps (this is a simplified version, a more detailed analysis is provided in Appendix G). This is reasonable for object detections but not for the estimations from KF. This is because KF’s estimations always rely on its estimations on previous time steps. The effect is usually minor because KF can use observation in update to prevent the posteriori state estimation and covariance, i.e. 𝐱^t|t\hat{\mathbf{x}}_{t|t} and 𝐏t|t\mathbf{P}_{t|t}, deviating from the true value too far away. However, when no observations are provided to KF, it cannot use observation to update its parameters. Then it has to follow Eq. 3 to prolong the estimated trajectory to the next time step. Consider a track is occluded on the time steps between tt and t+Tt+T and the noise of speed estimate follows δu˙t∼𝒩⁡(0,2​σu2)\delta_{\dot{u}_{t}}\sim\mathcal{N}(0,{2\sigma_{u}^{2}}), δv˙t∼𝒩⁡(0,2​σv2)\delta_{\dot{v}_{t}}\sim\mathcal{N}(0,{2\sigma_{v}^{2}}) for SORT. On the step t+Tt+T, state estimation would be

|  | ut+T=ut+T​u˙t,vt+T=vt+T​v˙t,u_{t+T}=u_{t}+T\dot{u}_{t},\quad\quad v_{t+T}=v_{t}+T\dot{v}_{t},\vskip-2.84544pt |  | (5) |
|---|---|---|---|

whose noise follows δut+T∼𝒩⁡(0,2​T2​σu2)\delta_{u_{t+T}}\sim\mathcal{N}(0,2T^{2}\sigma_{u}^{2}) and δvt+T∼𝒩⁡(0,2​T2​σv2)\delta_{v_{t+T}}\sim\mathcal{N}(0,2T^{2}\sigma_{v}^{2}). So without the observations, the estimation from the linear motion assumption of KF results in a fast error accumulation with respect to time. Given σv\sigma_{v} and σu\sigma_{u} is of the same magnitude as object displacement between consecutive frames, the noise of final object position (ut+T,vt+T)(u_{t+T},v_{t+T}) is of the same magnitude as the object size. For instance, the size of pedestrians close to the camera on MOT17 is around 50×30050\times 300 pixels. So even assuming the variance of position estimation is only 1 pixel, 10-frame occlusion can accumulate a shift in final position estimation as large as the object size. Such error magnification leads to a major accumulation of errors when the scenes are crowded.

The aforementioned limitations come from a fundamental property of SORT that it follows KF to be estimation-centric. It allows update without the existence of observations and purely trusts the estimations. A key difference between state estimations and observations is that we can assume that the observations by an object detector in each frame are affected by i.i.d. noise δ𝐳∼𝒩⁡(0,σ′2)\delta_{\mathbf{z}}\sim\mathcal{N}(0,{\sigma^{\prime}}^{2}) while the noise in state estimations can be accumulated along the hidden Markov process. Moreover, modern object detectors use powerful object visual features [51, 48]. It makes that, even on a single frame, it is usually safe to assume σ′<σu\sigma^{\prime}<\sigma_{u} and σ′<σv\sigma^{\prime}<\sigma_{v} because the object localization is more accurate by detection than from the state estimations through linear motion assumption. Combined with the previously mentioned two limitations, being estimation-centric makes SORT suffer from heavy noise when there is occlusion and the object motion is not perfectly linear.

## 4 Observation-Centric SORT

In this section, we introduce the proposed Observation-Centric SORT (OC-SORT). To address the limitations of SORT discussed above, we use the momentum of the object moving into the association stage and develop a pipeline with less noise and more robustness over occlusion and non-linear motion. The key is to design the tracker as observation-centric instead of estimation-centric. If a track is recovered from being untracked, we use an Observation-centric Re-Update (ORU) strategy to counter the accumulated error during the untracked period. OC-SORT also adds an Observation-Centric Momentum (OCM) term in the association cost. Please refer to Algorithm 1 in Appendix for the pseudo-code of OC-SORT. The pipeline is shown in Fig. 2.

### 4.1 Observation-centric Re-Update (ORU)

In practice, even if an object can be associated again by SORT after a period of being untracked, it is probably lost again because its KF parameters have already deviated far away from the correct due to the temporal error magnification. To alleviate this problem, we propose Observation-centric Re-Update (ORU) to reduce the accumulated error. Once a track is associated with an observation again after a period of being untracked (“re-activation”), we backcheck the period of its being lost and re-update the parameters of KF. The re-update is based on “observations” from a virtual trajectory. The virtual trajectory is generated referring to the observations on the steps starting and ending the untracked period. For example, by denoting the last-seen observation before being untracked as 𝐳t1\mathbf{z}_{t_{1}} and the observation triggering the re-association as 𝐳t2\mathbf{z}_{t_{2}}, the virtual trajectory is denoted as

|  | 𝐳~t=T​r​a​jvirtual​(𝐳t1,𝐳t2,t),t1<t<t2.\tilde{\mathbf{z}}_{t}=Traj_{\text{virtual}}(\mathbf{z}_{t_{1}},\mathbf{z}_{t_{2}},t),t_{1}<t<t_{2}. |  | (6) |
|---|---|---|---|

Then, along the trajectory of 𝐳~t​(t1<t<t2)\tilde{\mathbf{z}}_{t}(t_{1}<t<t_{2}), we run the loop of predict and re-update. The re-update operation is

|  | re-update{𝐊t=𝐏t|t−1​𝐇t⊤​(𝐇t​𝐏t|t−1​𝐇t⊤+𝐑t)−1𝐱^t|t=𝐱^t|t−1+𝐊t​(𝐳~t−𝐇t​𝐱^t|t−1)𝐏t|t=(𝐈−𝐊t​𝐇t)​𝐏t|t−1\text{\text{{re-update}}}\left\{\begin{aligned} \mathbf{K}_{t}&=\mathbf{P}_{t|t-1}\mathbf{H}_{t}^{\top}(\mathbf{H}_{t}\mathbf{P}_{t|t-1}\mathbf{H}_{t}^{\top}+\mathbf{R}_{t})^{-1}\\ \hat{\mathbf{x}}_{t|t}&=\hat{\mathbf{x}}_{t|t-1}+\mathbf{K}_{t}(\tilde{\mathbf{z}}_{t}-\mathbf{H}_{t}\hat{\mathbf{x}}_{t|t-1})\\ \mathbf{P}_{t|t}&=(\mathbf{I}-\mathbf{K}_{t}\mathbf{H}_{t})\mathbf{P}_{t|t-1}\end{aligned}\right. |  | (7) |
|---|---|---|---|

As the observations on the virtual trajectory match the motion pattern anchored by the last-seen and the latest association real observations, the update will not suffer from the error accumulated through the dummy update anymore. We call the proposed process Observation-centric Re-Update. It serves as an independent stage outside the predict-update loop and is triggered only a track is re-activated from a period of having no observations.

### 4.2 Observation-Centric Momentum (OCM)

In a reasonably short time interval, we can approximate the motion as linear. And the linear motion assumption also asks for consistent motion direction. But the noise prevents us from leveraging the consistency of direction. To be precise, to determine the motion direction, we need the object state on two steps with a time difference Δ​t\Delta t. If Δ​t\Delta t is small, the velocity noise would be significant because of the estimation’s sensitivity to state noise. If Δ​t\Delta t is big, the noise of direction estimation can also be significant because of the temporal error magnification and the failure of linear motion assumption. As state observations have no problem of temporal error magnification that state estimations suffer from, we propose to use observations instead of estimations to reduce the noise of motion direction calculation and introduce the term of its consistency to help the association.

With the new term, given NN existing tracks and MM detections on the new-coming time step, the association cost matrix is formulated as

|  | C⁡(𝐗^,𝐙)=CIoU​(𝐗^,𝐙)+λ​Cv​(𝒵,𝐙),C(\hat{\mathbf{X}},\mathbf{Z})=C_{\text{IoU}}(\hat{\mathbf{X}},\mathbf{Z})+\lambda C_{v}(\mathcal{Z},\mathbf{Z}), |  | (8) |
|---|---|---|---|

where 𝐗^∈ℝN×7\hat{\mathbf{X}}\in\mathbb{R}^{N\times 7} is the set of object state estimations and 𝐙∈ℝM×5\mathbf{Z}\in\mathbb{R}^{M\times 5} is the set of observations on the new time step. λ\lambda is a weighting factor. 𝒵\mathcal{Z} contains the trajectory of observations of all existing tracks. CIoU​(⋅,⋅)C_{\text{IoU}}(\cdot,\cdot) calculates the negative pairwise IoU (Intersection over Union) and Cv​(⋅,⋅)C_{v}(\cdot,\cdot) calculates the consistency between the directions of i) linking two observations on an existing track (θtrack\theta^{\text{track}}) and ii) linking a track’s historical observation and a new observation (θintention\theta^{\text{intention}}). CvC_{v} contains all pairs of Δ​θ=|θtrack−θintention|\Delta\theta=|\theta^{\text{track}}-\theta^{\text{intention}}|. In our implementation, we calculate the motion direction in radians, namely θ=arctan⁡(v1−v2u1−u2)\theta=\arctan(\frac{v_{1}-v_{2}}{u_{1}-u_{2}}) where (u1,v1)(u_{1},v_{1}) and (u2,v2)(u_{2},v_{2}) are the observations on two different time steps. The calculation of this is also illustrated in Figure 4.

Following the assumptions of noise distribution mentioned before, we can derive a closed-form probability density function of the distribution of the noise in the direction estimation. The derivation is explained in detail in Appendix A. By analyzing the property of this distribution, we reach a conclusion that, under the linear-motion model, the scale of the noise of direction estimation is negatively correlated to the time difference between the two observation points, i.e. Δ​t\Delta t. This suggests increasing Δ​t\Delta t to achieve a low-noisy estimation of θ\theta. However, the assumption of linear motion typically holds only when Δ​t\Delta t is small enough. Therefore, the choice of Δ​t\Delta t requires a trade-off.

Besides ORU and OCM, we also find it empirically helpful to check a track’s last presence to recover it from being lost. We thus apply a heuristic Observation-Centric Recovery (OCR) technique. OCR will start a second attempt of associating between the last observation of unmatched tracks to the unmatched observations after the usual association stage. It can handle the case of an object stopping or being occluded for a short time interval.

## 5 Experiments

### 5.1 Experimental Setup

Datasets. We evaluate our method on multiple multi-object tracking datasets including MOT17 [41], MOT20 [14], KITTI [20], DanceTrack [54] and CroHD [56]. MOT17 [41] and MOT20 [14] are for pedestrian tracking, where targets mostly move linearly, while scenes in MOT20 are more crowded. KITTI [20] is for pedestrian and car tracking with a relatively low frame rate of 1010FPS. CroHD is a dataset for head tracking in the crowd and the results on it are included in the appendix. DanceTrack [54] is a recently proposed dataset for human tracking. For the data in DanceTrack, object localization is easy, but the object motion is highly non-linear. Furthermore, the objects have a close appearance, severe occlusion, and frequent crossovers. Considering our goal is to improve tracking robustness under occlusion and non-linear object motion, we would emphasize the comparison on DanceTrack.

| Tracker | HOTA↑\uparrow | MOTA↑\uparrow | IDF1↑\uparrow | FP(10410^{4})↓\downarrow | FN(10410^{4})↓\downarrow | IDs↓\downarrow | Frag↓\downarrow | AssA↑\uparrow | AssR↑\uparrow |
|---|---|---|---|---|---|---|---|---|---|
| FairMOT [71] | 59.3 | 73.7 | 72.3 | 2.75 | 11.7 | 3,303 | 8,073 | 58.0 | 63.6 |
| TransCt [67] | 54.5 | 73.2 | 62.2 | 2.31 | 12.4 | 4,614 | 9,519 | 49.7 | 54.2 |
| TransTrk [55] | 54.1 | 75.2 | 63.5 | 5.02 | 8.64 | 3,603 | 4,872 | 47.9 | 57.1 |
| GRTU [60] | 62.0 | 74.9 | 75.0 | 3.20 | 10.8 | 1,812 | 1,824 | 62.1 | 65.8 |
| QDTrack [42] | 53.9 | 68.7 | 66.3 | 2.66 | 14.7 | 3,378 | 8,091 | 52.7 | 57.2 |
| MOTR [69] | 57.2 | 71.9 | 68.4 | 2.11 | 13.6 | 2,115 | 3,897 | 55.8 | 59.2 |
| PermaTr [57] | 55.5 | 73.8 | 68.9 | 2.90 | 11.5 | 3,699 | 6,132 | 53.1 | 59.8 |
| TransMOT [12] | 61.7 | 76.7 | 75.1 | 3.62 | 9.32 | 2,346 | 7,719 | 59.9 | 66.5 |
| GTR [75] | 59.1 | 75.3 | 71.5 | 2.68 | 11.0 | 2,859 | - | 61.6 | - |
| DST-Tracker [8] | 60.1 | 75.2 | 72.3 | 2.42 | 11.0 | 2,729 | - | 62.1 | - |
| MeMOT [5] | 56.9 | 72.5 | 69.0 | 2.72 | 11.5 | 2,724 | - | 55.2 | - |
| UniCorn [68] | 61.7 | 77.2 | 75.5 | 5.01 | 7.33 | 5,379 | - | - | - |
| ByteTrack [70] | 63.1 | 80.3 | 77.3 | 2.55 | 8.37 | 2,196 | 2,277 | 62.0 | 68.2 |
| OC-SORT | 63.2 | 78.0 | 77.5 | 1.51 | 10.8 | 1,950 | 2,040 | 63.2 | 67.5 |
|  |  |  |  |  |  |  |  |  |  |

| Tracker | HOTA↑\uparrow | MOTA↑\uparrow | IDF1↑\uparrow | FP(10410^{4})↓\downarrow | FN(10410^{4})↓\downarrow | IDs↓\downarrow | Frag↓\downarrow | AssA↑\uparrow | AssR↑\uparrow |
|---|---|---|---|---|---|---|---|---|---|
| FairMOT [71] | 54.6 | 61.8 | 67.3 | 10.3 | 8.89 | 5,243 | 7,874 | 54.7 | 60.7 |
| TransCt [67] | 43.5 | 58.5 | 49.6 | 6.42 | 14.6 | 4,695 | 9,581 | 37.0 | 45.1 |
| Semi-TCL [35] | 55.3 | 65.2 | 70.1 | 6.12 | 11.5 | 4,139 | 8,508 | 56.3 | 60.9 |
| CSTrack [36] | 54.0 | 66.6 | 68.6 | 2.54 | 14.4 | 3,196 | 7,632 | 54.0 | 57.6 |
| GSDT [61] | 53.6 | 67.1 | 67.5 | 3.19 | 13.5 | 3,131 | 9,875 | 52.7 | 58.5 |
| TransMOT [12] | 61.9 | 77.5 | 75.2 | 3.42 | 8.08 | 1,615 | 2,421 | 60.1 | 66.3 |
| MeMOT [5] | 54.1 | 63.7 | 66.1 | 4.79 | 13.8 | 1,938 | - | 55.0 | - |
| ByteTrack [70] | 61.3 | 77.8 | 75.2 | 2.62 | 8.76 | 1,223 | 1,460 | 59.6 | 66.2 |
| OC-SORT | 62.1 | 75.5 | 75.9 | 1.80 | 10.8 | 913 | 1,198 | 62.0 | 67.5 |
|  |  |  |  |  |  |  |  |  |  |

| Tracker | HOTA↑\uparrow | DetA↑\uparrow | AssA↑\uparrow | MOTA↑\uparrow | IDF1↑\uparrow |
|---|---|---|---|---|---|
| CenterTrack [73] | 41.8 | 78.1 | 22.6 | 86.8 | 35.7 |
| FairMOT [71] | 39.7 | 66.7 | 23.8 | 82.2 | 40.8 |
| QDTrack [42] | 45.7 | 72.1 | 29.2 | 83.0 | 44.8 |
| TransTrk[55] | 45.5 | 75.9 | 27.5 | 88.4 | 45.2 |
| TraDes [64] | 43.3 | 74.5 | 25.4 | 86.2 | 41.2 |
| MOTR [69] | 54.2 | 73.5 | 40.2 | 79.7 | 51.5 |
| GTR [75] | 48.0 | 72.5 | 31.9 | 84.7 | 50.3 |
| DST-Tracker [8] | 51.9 | 72.3 | 34.6 | 84.9 | 51.0 |
| SORT [3] | 47.9 | 72.0 | 31.2 | 91.8 | 50.8 |
| DeepSORT [63] | 45.6 | 71.0 | 29.7 | 87.8 | 47.9 |
| ByteTrack [70] | 47.3 | 71.6 | 31.4 | 89.5 | 52.5 |
| OC-SORT | 54.6 | 80.4 | 40.2 | 89.6 | 54.6 |
| OC-SORT + Linear Interp | 55.1 | 80.4 | 40.4 | 92.2 | 54.9 |
|  |  |  |  |  |  |

|  | Car | Pedestrian |  |  |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|
| Tracker | HOTA↑\uparrow | MOTA↑\uparrow | AssA↑\uparrow | IDs↓\downarrow | Frag↓\downarrow | HOTA↑\uparrow | MOTA↑\uparrow | AssA↑\uparrow | IDs↓\downarrow | Frag↓\downarrow |
| IMMDP [65] | 68.66 | 82.75 | 69.76 | 211 | 181 | - | - | - | - | - |
| SMAT [21] | 71.88 | 83.64 | 72.13 | 198 | 294 | - | - | - | - | - |
| TrackMPNN [45] | 72.30 | 87.33 | 70.63 | 481 | 237 | 39.40 | 52.10 | 35.45 | 626 | 669 |
| MPNTrack [4] | - | - | - | - | - | 45.26 | 46.23 | 47.28 | 397 | 1,078 |
| CenterTr [73] | 73.02 | 88.83 | 71.18 | 254 | 227 | 40.35 | 53.84 | 36.93 | 425 | 618 |
| LGM [59] | 73.14 | 87.60 | 72.31 | 448 | 164 | - | - | - | - | - |
| TuSimple [11] | 71.55 | 86.31 | 71.11 | 292 | 218 | 45.88 | 57.61 | 47.62 | 246 | 651 |
| PermaTr [57] | 77.42 | 90.85 | 77.66 | 275 | 271 | 47.43 | 65.05 | 43.66 | 483 | 703 |
| OC-SORT | 74.64 | 87.81 | 74.52 | 257 | 318 | 52.95 | 62.00 | 57.81 | 181 | 598 |
| OC-SORT + HP | 76.54 | 90.28 | 76.39 | 250 | 280 | 54.69 | 65.14 | 59.08 | 184 | 609 |
|  |  |  |  |  |  |  |  |  |  |  |

Implementations. For a fair comparison, we directly apply the object detections from existing baselines. For MOT17, MOT20, and DanceTrack, we use the publicly available YOLOX [19] detector weights by ByteTrack [70]. For KITTI [20], we use the detections from PermaTrack [57] publicly available in the official release11 1 https://github.com/TRI-ML/permatrack/. For ORU, we generate the virtual trajectory during occlusion with the constant-velocity assumption. Therefore, Eq. 6 is adopted as 𝐳~t=𝐳t1+t−t1t2−t1​(𝐳t2−𝐳t1),t1<t<t2\tilde{\mathbf{z}}_{t}=\mathbf{z}_{t_{1}}+\frac{t-t_{1}}{t_{2}-t_{1}}(\mathbf{z}_{t_{2}}-\mathbf{z}_{t_{1}}),t_{1}<t<t_{2}. For OCM, the velocity direction is calculated using the observations three time steps apart, i.e. Δ​t=3\Delta t=3. The direction difference is measured by the absolute difference of angles in radians. We set λ=0.2\lambda=0.2 in Eq. 8. Following the common practice of SORT, we set the detection confidence threshold at 0.40.4 for MOT20 and 0.60.6 for other datasets. The IoU threshold during association is 0.30.3.

Metrics. We adopt HOTA [37] as the main metric as it maintains a better balance between the accuracy of object detection and association [37]. We also emphasize AssA to evaluate the association performance. IDF1 is also used for association performance evaluation. Other metrics we report, such as MOTA, are highly related to detection performance. It is fair to use these metrics only when all methods use the same detections for tracking, which is referred to as “public tracking” as reported in Appendix C.

### 5.2 Benchmark Results

Here we report the benchmark results on multiple datasets. We put all methods that use the shared detection results in a block at the bottom of each table.

|  | MOT17-val | DanceTrack-val |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|
| ORU | OCM | OCR | HOTA↑\uparrow | AssA↑\uparrow | IDF1↑\uparrow | HOTA↑\uparrow | AssA↑\uparrow | IDF1↑\uparrow |
|  |  |  | 64.9 | 66.8 | 76.9 | 47.8 | 31.0 | 48.3 |
| ✓ |  |  | 66.3 | 68.0 | 77.2 | 48.5 | 32.2 | 49.8 |
| ✓ | ✓ |  | 66.4 | 69.0 | 77.8 | 52.1 | 35.0 | 50.6 |
| ✓ | ✓ | ✓ | 66.5 | 68.9 | 77.7 | 52.1 | 35.3 | 51.6 |

|  | MOT17-val | DanceTrack-val |  |  |  |  |
|---|---|---|---|---|---|---|
|  | HOTA↑\uparrow | AssA↑\uparrow | IDF1↑\uparrow | HOTA↑\uparrow | AssA↑\uparrow | IDF1↑\uparrow |
| Const. Speed | 66.5 | 68.9 | 77.7 | 52.1 | 35.3 | 51.6 |
| GPR | 63.1 | 65.2 | 75.7 | 49.5 | 33.7 | 49.6 |
| Linear Regression | 64.3 | 66.5 | 76.0 | 49.3 | 33.4 | 49.2 |
| Const. Acceleration | 66.2 | 67.9 | 77.4 | 51.3 | 34.8 | 50.9 |

|  | MOT17-val | DanceTrack-val |  |  |  |  |
|---|---|---|---|---|---|---|
|  | HOTA↑\uparrow | AssA↑\uparrow | IDF1↑\uparrow | HOTA↑\uparrow | AssA↑\uparrow | IDF1↑\uparrow |
| Δ​t=1\Delta t=1 | 66.1 | 67.5 | 76.9 | 51.3 | 34.3 | 51.3 |
| Δ​t=2\Delta t=2 | 66.3 | 68.0 | 77.3 | 52.2 | 35.4 | 51.4 |
| Δ​t=3\Delta t=3 | 66.5 | 68.9 | 77.7 | 52.1 | 35.3 | 51.6 |
| Δ​t=6\Delta t=6 | 66.0 | 67.5 | 76.9 | 52.1 | 35.4 | 51.8 |

MOT17 and MOT20. We report OC-SORT’s performance on MOT17 and MOT20 in Table 1 and Table 2 using private detections. To make a fair comparison, we use the same detection as ByteTrack [70]. OC-SORT achieves performance comparable to other state-of-the-art methods. Our gains are especially significant in MOT20 under severe pedestrian occlusion, setting a state-of-the-art HOTA of 62.162.1. As our method is designed to be simple for better generalization, we do not use adaptive detection thresholds as in ByteTrack. Also, ByteTrack uses more detections of low-confidence to achieve higher MOTA scores but we keep the detection confidence threshold the same as on other datasets, which is the common practice in the community. We inherit the linear interpolation on the two datasets as baseline methods for a fair comparison. To more clearly discard the variance from the detector, we also perform public tracking on MOT17 and MOT20, which is reported in Table 12 and Table 13 in Appendix C. OC-SORT still outperforms the existing state-of-the-art in public tracking settings.

DanceTrack. To evaluate OC-SORT under challenging non-linear object motion, we report results on the DanceTrack in Table 3. OC-SORT sets a new state-of-the-art, outperforming the baselines by a great margin under non-linear object motions. We compare the tracking results of SORT and OC-SORT under extreme non-linear situations in Fig.1 and more samples are available in Fig. 8 in Appendix E. We also visualize the output trajectories by OC-SORT and SORT on randomly selected DanceTrack video clips in Fig. 9 in Appendix E. For multi-object tracking in occlusion and non-linear motion, the results on DanceTrack are strong evidence of the effectiveness of OC-SORT.

KITTI. In Table 4 we report the results on the KITTI dataset. For a fair comparison, we adopt the detector weights by PermaTr [57] and report its performance in the table as well. Then, we run OC-SORT given the shared detections. As initializing SORT’s track requires continuous tracking across several frames (“minimum hits”), we observe that the results not recorded during the track initialization make a significant difference. To address this problem, we perform offline head padding (HP) post-processing by writing these entries back after finishing the online tracking stage. The results of the car category on KITTI show an essential shortcoming of the default implementation version of OC-SORT that it chooses the IoU matching for the association. When the object velocity is high or the frame rate is low, the IoU of object bounding boxes between consecutive frames can be very low or even zero. This issue does not come from the intrinsic design of OC-SORT and is widely observed when using IoU as the association cue. Adding other cues [72, 49, 73] and appearance similarity [63, 38] have been demonstrated [63] efficient to solve this. In contrast to the relatively inferior car tracking performance, OC-SORT improves pedestrian tracking performance to a new state-of-the-art. Using the same detections, OC-SORT achieves a large performance gap over PermaTr with 10x faster speed.

The results on multiple benchmarks have demonstrated the effectiveness and efficiency of OC-SORT. We note that we use a shared parameter stack across datasets. Carefully tuning the parameters can probably further boost the performance. For example, the adaptive detection threshold is proven useful in previous work [70]. Besides the association accuracy, we also care about the inference speed. Given off-the-shelf detections, OC-SORT runs at 793793 FPS on an Intel i9-9980XE CPU @ 3.00GHz. Therefore, OC-SORT can still run in an online and real-time fashion.

### 5.3 Ablation Study

Component Ablation. We ablate the contribution of proposed modules on the validation sets of MOT17 and DanceTrack in Table 5. The splitting of the MOT17 validation set follows a popular convention [73]. The results demonstrate the efficiency of the proposed modules in OC-SORT. The results show that the performance gain from ORU is significant on both datasets but OCM only shows good help on DanceTrack dataset where object motion is more complicated and the occlusion is heavy. It suggests the effectiveness of our proposed method to improve tracking robustness in occlusion and non-linear motion.

Virtual Trajectory in ORU. For simplicity, we follow the naive hypothesis of constant speed to generate a virtual trajectory in ORU. There are other alternatives like constant acceleration, regression-based fitting such as Linear Regression (LR) or Gaussian Process Regression (GPR), and Near Constant Acceleration Model (NCAM) [27]. The results of comparing these choices are shown in Table 6. For GPR, we use the RBF kernel [10] k⁡(𝐱,𝐱′)=exp​(−‖𝐱−𝐱′‖250)k(\mathbf{x},\mathbf{x}^{\prime})=\text{exp}\left(-\frac{||\mathbf{x}-\mathbf{x}^{\prime}||^{2}}{50}\right). We provide more studies on the kernel configuration in Appendix B. The results show that local hypotheses such as Constant Speed/Acceleration perform much better than global hypotheses such as LR and GPR. This is probably because, as virtual trajectory generation happens in an online fashion, it is hard to get a reliable fit using only limited data points on historical time steps.

Δ​t\Delta t in OCM. There is a trade-off when choosing the time difference Δ​t\Delta t in OCM (Section 4). A large Δ​t\Delta t decreases the noise of velocity estimation. but is also likely to discourage approximating object motion as linear. Therefore, we study the influence of varying Δ​t\Delta t in Table 7. Our results agree with our analysis that increasing Δ​t\Delta t from Δ​t=1\Delta t=1 can boost the association performance. But increasing Δ​t\Delta t higher than the bottleneck instead hurts the performance because of the difficulty of maintaining the approximation of linear motion.

## 6 Conclusion

We analyze the popular motion-based tracker SORT and recognize its intrinsic limitations from using Kalman filter. These limitations significantly hurt tracking accuracy when the tracker fails to gain observations for supervision - likely caused by unreliable detectors, occlusion, or fast and non-linear target object motion. To address these issues, we propose Observation-Centric SORT (OC-SORT). OC-SORT is more robust to occlusion and non-linear object motion while keeping simple, online, and real-time. In our experiments on diverse datasets, OC-SORT significantly outperforms the state-of-the-art. The gain is especially significant for multi-object tracking under occlusion and non-linear object motion.

Acknowledgement. We thank David Held, Deva Ramanan, and Siddharth Ancha for the discussion about the theoretical modeling of OC-SORT. We thank Yuda Song for the help with the analysis of OCM error distribution. We also would like to thank Zhengyi Luo, Yuda Song, and Erica Weng for the detailed feedback on the paper writing. We also thank Yifu Zhang for sharing his experience with ByteTrack. This project was sponsored in part by NSF NRI award 2024173.

## Appendix A Velocity Direction Variance in OCM

In this section, we work on the setting of linear motion with noisy states. We provide proof that the trajectory direction estimation has a smaller variance if the two states we use for the estimation have a larger time difference. We assume the motion model is 𝐱t=f⁡(t)+ϵ\mathbf{x}_{t}=f(t)+\epsilon where ϵ\epsilon is gaussian noise and the ground-truth center position of the target is (μut,μvt)(\mu_{u_{t}},\mu_{v_{t}}) at time step tt. Then the true motion direction between the two time steps is

|  | θ=arctan⁡(μvt1−μvt2μut1−μut2).\theta=\arctan(\frac{\mu_{v_{t_{1}}}-\mu_{v_{t_{2}}}}{\mu_{u_{t_{1}}}-\mu_{u_{t_{2}}}}). |  | (9) |
|---|---|---|---|

And we have |μvt1−μvt2|∝|t1−t2||\mu_{v_{t_{1}}}-\mu_{v_{t_{2}}}|\propto|t_{1}-t_{2}|, |μut1−μut2|∝|t1−t2||\mu_{u_{t_{1}}}-\mu_{u_{t_{2}}}|\propto|t_{1}-t_{2}|. As the detection results do not suffer from the error accumulation due to propagating along Markov process as Kalman filter does, we can assume the states from observation suffers some i.i.d. noise, i.e., ut∼𝒩⁡(μut,σu2)u_{t}\sim\mathcal{N}(\mu_{u_{t}},\sigma_{u}^{2}) and vt∼𝒩⁡(μvt,σv2)v_{t}\sim\mathcal{N}(\mu_{v_{t}},\sigma_{v}^{2}). We now analyze the noise of the estimated θ~=vt1−vt2ut1−ut2\tilde{\theta}=\frac{v_{t_{1}}-v_{t_{2}}}{u_{t_{1}}-u_{t_{2}}} by two observations on the trajectory. Because the function of arctan⁡(⋅)\arctan(\cdot) is monotone over the whole real field, we can study tan⁡θ~\tan\tilde{\theta} instead which simplifies the analysis. We denote w=ut1−ut2w=u_{t_{1}}-u_{t_{2}}, y=vt1−vt2y=v_{t_{1}}-v_{t_{2}}, and z=ywz=\frac{y}{w}, first we can see that yy and ww jointly form a Gaussian distribution:

|  | [yw]∼𝒩⁡([μyμw],[σy2ρ​σy​σwρ​σy​σwσw2]),\begin{bmatrix}y\\ w\end{bmatrix}\sim\mathcal{N}\left(\begin{bmatrix}\mu_{y}\\ \mu_{w}\end{bmatrix},\begin{bmatrix}\sigma_{y}^{2}&\rho\sigma_{y}\sigma_{w}\\ \rho\sigma_{y}\sigma_{w}&\sigma_{w}^{2}\end{bmatrix}\right), |  | (10) |
|---|---|---|---|

where μy=μvt1−μvt​2\mu_{y}=\mu_{v_{t_{1}}}-\mu_{v_{t2}}, μw=μut1−μut2\mu_{w}=\mu_{u_{t_{1}}}-\mu_{u_{t_{2}}}, σw=2​σu\sigma_{w}=\sqrt{2}\sigma_{u} and σy=2​σv\sigma_{y}=\sqrt{2}\sigma_{v}, and ρ\rho is the correlation coefficient between yy and ww. We can derive a closed-form solution of the probability density function [24] of zz as

|  | p⁡(z)=\displaystyle p(z)= | g⁡(z)​eg​(z)2−α​r​(z)22​β2​r​(z)22​π​σw​σy​r​(z)3​[Φ⁡(g⁡(z)β​r​(z))−Φ⁡(−g⁡(z)β​r​(z))]\displaystyle\frac{g(z)e^{\frac{g(z)^{2}-\alpha r(z)^{2}}{2\beta^{2}r(z)^{2}}}}{\sqrt{2\pi}\sigma_{w}\sigma_{y}r(z)^{3}}\left[\Phi\left(\frac{g(z)}{\beta r(z)}\right)-\Phi\left(-\frac{g(z)}{\beta r(z)}\right)\right] |  | (11) |
|---|---|---|---|---|
|  |  | +βe−2α/βπ​σw​σy​r​(z)2\displaystyle+\frac{\beta e^{-2\alpha/\beta}}{\pi\sigma_{w}\sigma_{y}r(z)^{2}} |  |  |

where

|  | r⁡(z)\displaystyle r(z) | =z2σy2−2​ρ​zσy​σw+1σw2,\displaystyle=\sqrt{\frac{z^{2}}{\sigma_{y}^{2}}-\frac{2\rho z}{\sigma_{y}\sigma_{w}}+\frac{1}{\sigma_{w}^{2}}}, |  | (12) |
|---|---|---|---|---|
|  | g⁡(z)\displaystyle g(z) | =μy​zσy2−ρ⁡(μy+μw​z)σy​σw+μwσw2,\displaystyle=\frac{\mu_{y}z}{\sigma_{y}^{2}}-\frac{\rho(\mu_{y}+\mu_{w}z)}{\sigma_{y}\sigma_{w}}+\frac{\mu_{w}}{\sigma_{w}^{2}}, |  |  |
|  | α\displaystyle\alpha | =μw2+μy2σy2−2​ρ​μy​μwσw​σy,β=1−ρ2,\displaystyle=\frac{\mu_{w}^{2}+\mu_{y}^{2}}{\sigma_{y}^{2}}-\frac{2\rho\mu_{y}\mu_{w}}{\sigma_{w}\sigma_{y}},\quad\quad\beta=\sqrt{1-\rho^{2}}, |  |  |

and Φ\Phi is the cumulative distribution function of the standard normal. Without loss of generality, we can assume μw>0\mu_{w}>0 and μy>0\mu_{y}>0 because negative ground-truth displacements enjoy the same property. This solution has a good property that larger μw\mu_{w} or μy\mu_{y} makes the probability density at the true value, i.e. μz=μyμw\mu_{z}=\frac{\mu_{y}}{\mu_{w}}, higher, and the tails decay more rapidly. So the estimation of arctan⁡θ\arctan\theta, also θ\theta, has smaller noise when μw\mu_{w} or μy\mu_{y} is larger. Under the assumption of linear motion, we thus should select two observations with a large temporal difference to estimate the direction.

It is reasonable to assume the noise of detection along the u-axis and v-axis are independent so ρ=0\rho=0. And when representing the center position in pixel, it is also moderate to assume σw=σy=1\sigma_{w}=\sigma_{y}=1 (also for the ease of presentation). Then, with different true value of μz=μyμw\mu_{z}=\frac{\mu_{y}}{\mu_{w}}, the visualizations of p⁡(z)p(z) over zz and μy\mu_{y} are shown in Figure 5. The visualization demonstrates our analysis above. Moreover, it shows that when the value of μy\mu_{y} or μw\mu_{w} is small, the cluster peak of the distribution at μz\mu_{z} is not significant anymore, as the noise σy\sigma_{y} and σw\sigma_{w} can be dominant. Considering the visualization shows that happens when μy\mu_{y} is close to σy\sigma_{y}, this can happen when we estimate the speed by observations from two consecutive frames because the variance of observation can be close to the absolute displacement of object motion. This makes another support to our analysis in the main paper about the sensitivity to state estimation noise.

|  | MOT17-val | DanceTrack-val |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|
|  | HOTA↑\uparrow | AssA↑\uparrow | MOTA↑\uparrow | IDF1↑\uparrow | HOTA↑\uparrow | AssA↑\uparrow | MOTA↑\uparrow | IDF1↑\uparrow |
| w/o interpolation | 66.5 | 68.9 | 74.9 | 77.7 | 52.1 | 35.3 | 87.3 | 51.6 |
| Linear Interpolation | 68.0 | 69.9 | 77.9 | 79.3 | 52.8 | 35.6 | 89.8 | 52.1 |
| GPR Interpolation | 65.2 | 67.0 | 72.9 | 75.9 | 51.6 | 35.0 | 86.1 | 51.2 |

|  | MOT17-val | DanceTrack-val |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|
| Interpolation Method | HOTA | AssA | MOTA | IDF1 | HOTA | AssA | MOTA | IDF1 |
| w/o interpolation | 66.5 | 68.9 | 74.9 | 77.7 | 52.1 | 35.3 | 87.3 | 51.6 |
| Linear Interpolation | 69.6 | 69.9 | 77.9 | 79.3 | 52.8 | 35.6 | 89.8 | 52.1 |
| GPR Interp, l=1l=1 | 66.2 | 67.6 | 74.3 | 76.6 | 51.8 | 35.0 | 86.6 | 50.8 |
| GPR Interp, l=5l=5 | 66.3 | 67.0 | 72.9 | 75.9 | 51.8 | 35.1 | 86.5 | 51.1 |
| GPR Interp, l=Lτl=L_{\tau} | 66.1 | 67.0 | 73.1 | 77.8 | 51.6 | 35.1 | 86.4 | 50.7 |
| GPR Interp, l=1000/Lτl=1000/L_{\tau} | 65.9 | 67.0 | 73.0 | 77.8 | 51.8 | 35.0 | 86.9 | 51.0 |
| GPR Interp, l=l= MT(τ\tau) | 65.9 | 67.0 | 73.1 | 77.8 | 51.7 | 35.1 | 86.7 | 50.9 |
| LI + GPR Smoothing, l=1l=1 | 69.5 | 69.6 | 77.8 | 79.3 | 52.8 | 35.6 | 89.9 | 52.1 |
| LI + GPR Smoothing, l=5l=5 | 69.5 | 69.7 | 77.8 | 79.3 | 52.9 | 34.9 | 89.7 | 52.1 |
| LI + GPR Smoothing, l=Lτl=L_{\tau} | 69.6 | 69.5 | 77.8 | 79.2 | 52.9 | 35.6 | 89.9 | 52.1 |
| LI + GPR Smoothing, l=1000/Lτl=1000/L_{\tau} | 69.5 | 69.9 | 77.8 | 79.3 | 53.0 | 35.6 | 89.9 | 52.1 |
| LI + GPR Smoothing, l=MT​(τ)l=\text{MT}(\tau) | 69.5 | 69.6 | 77.8 | 79.3 | 52.8 | 35.6 | 89.8 | 52.1 |

## Appendix B Interpolation by Gaussian Progress Regression

Interpolation as post-processing. Although we focus on developing an online tracking algorithm, we are also interested in whether post-process can further optimize the tracking results in diverse conditions. Despite the failure of GPR in online tracking in Table 6, we continue to study if GPR is better suited for interpolation in Table 8. We compare GPR with the widely-used linear interpolation. The maximum gap for interpolation is set as 2020 frames and we use the same kernel for GPR as mentioned above. The results suggest that the GPR’s non-linear interpolation is simply not efficient. We think this is due to limited data points which results in an inaccurate fit of the object trajectory. Further, the variance in regressor predictions introduces extra noise. Although GPR interpolation decreases the performance on MOT17-val significantly, its negative influence on DanceTrack is relatively minor where the object motion is more non-linear. We believe how to fit object trajectory with non-linear hypothesis still requires more study.

From the analysis in the main paper, the failure of SORT can mainly result from occlusion (lack of observations) or the non-linear motion of objects (the break of the linear-motion assumption). So the question arises naturally whether we can extend SORT free of the linear-motion assumption or at least more robust when it breaks.

One way is to extend from KF to non-linear filters, such as EKF [30, 52] and UKF [28]. However, for real-world online tracking, they can be hard to be adopted as they need knowledge about the motion pattern or still rely on the techniques fragile to non-linear patterns, such as linearization [29]. Another choice is to gain the knowledge beyond linearity by regressing previous trajectory, such as combing Gaussian Process (GP) [62, 47, 32]: given a observation 𝐳⋆\mathbf{z}_{\star} and a kernel function k⁡(⋅,⋅)k(\cdot,\cdot), GP defines gaussian functions with mean μ𝐳⋆\mu_{\mathbf{z}_{\star}} and variance Σ𝐳⋆\Sigma_{\mathbf{z}_{\star}} as

|  |  | μ𝐳⋆=𝐤⋆⊤​[𝐊+σ2​𝐈]−1​𝐲,\displaystyle\mu_{\mathbf{z}_{\star}}=\mathbf{k}_{\star}^{\top}[\mathbf{K}+\sigma^{2}\mathbf{I}]^{-1}\mathbf{y}, |  | (13) |
|---|---|---|---|---|
|  |  | Σ𝐳⋆=k⁡(𝐳⋆,𝐳⋆)−𝐤⋆⊤​[𝐊+σ2​𝐈]−1​𝐤⋆,\displaystyle\Sigma_{\mathbf{z}_{\star}}=k(\mathbf{z}_{\star},\mathbf{z}_{\star})-\mathbf{k}_{\star}^{\top}[\mathbf{K}+\sigma^{2}\mathbf{I}]^{-1}\mathbf{k}_{\star}, |  |  |

where 𝐤⋆\mathbf{k}_{\star} is the kernel matrix between the input and training data and 𝐊\mathbf{K} is the kernel matrix over training data, 𝐲\mathbf{y} is the output of data. Until now, we have shown the primary study of using Gaussian Process Regression (GPR) in the online generation of the virtual trajectory in ORU and offline interpolation. But neither of them successfully boosts the tracking performance. Now, We continue to investigate in detail the chance of combining GPR and SORT for multi-object tracking for interpolation as some designs are worth more study.

### B.1 Choice of Kernel Function in Gaussian Process

The kernel function is a key variable of GPR. There is not a generally efficient guideline to choose the kernel for Gaussian Process Regression though some basic observations are available [15]. When there is no additional knowledge about the time sequential data to fit, the RBF kernel is one of the most common choices:

|  | k⁡(𝐱,𝐱′)=σ2​exp​(−‖𝐱−𝐱′‖22​l2),k(\mathbf{x},\mathbf{x}^{\prime})=\sigma^{2}\text{exp}\left(-\frac{||\mathbf{x}-\mathbf{x}^{\prime}||^{2}}{2l^{2}}\right), |  | (14) |
|---|---|---|---|

where ll is the lengthscale of the data to be fit. It determines the length of the “wiggles” of the target function. σ2\sigma^{2} is the output variance that determines the average distance of the function away from its mean. This is usually just a scale factor [15]. GPR is considered sensitive to ll in some situations. So we conduct an ablation study over it in the offline interpolation to see if we can use GPR to outperform the linear interpolation widely used in multi-object tracking.

| Tracker | HOTA↑\uparrow | MOTA↑\uparrow | IDF1↑\uparrow | FP(10410^{4})↓\downarrow | FN(10410^{4})↓\downarrow | IDs↓\downarrow | Frag↓\downarrow |
|---|---|---|---|---|---|---|---|
| HeadHunter [56] | 36.8 | 57.8 | 53.9 | 5.18 | 30.0 | 4,394 | 15,146 |
| HeadHunter dets + OC-SORT | 39.0 | 60.0 | 56.8 | 5.18 | 28.1 | 4,122 | 10,483 |
| FairMOT [71] | 43.0 | 60.8 | 62.8 | 11.8 | 19.9 | 12,781 | 41,399 |
| FairMOT dets + OC-SORT | 44.1 | 67.9 | 62.9 | 10.2 | 16.4 | 4,243 | 10,122 |

| Tracker | HOTA↑\uparrow | DetA↑\uparrow | AssA↑\uparrow | MOTA↑\uparrow | IDF1↑\uparrow |
|---|---|---|---|---|---|
| SORT | 47.9 | 72.0 | 31.2 | 91.8 | 50.8 |
| OC-SORT | 55.1 | 80.3 | 38.0 | 89.4 | 54.2 |
| OC-SORT (MOT17) | 48.6 | 71.0 | 33.3 | 84.2 | 51.5 |

### B.2 GPR for Offline Interpolation

We have presented the use of GPR in online virtual trajectory fitting and offline interpolation where we use l2=25l^{2}=25 and σ=1\sigma=1 for the kernel in Eq. 14. Further, we make a more thorough study of the setting of GPR. We follow the settings of experiments in the main paper that only trajectories longer than 30 frames are put into interpolation. And the interpolation is only applied to the gap shorter than 20 frames. We conduct the experiments on the validation sets of MOT17 and DanceTrack.

For the value of ll, we try fixed values, i.e. l=1l=1 and l=5l=5 (2​l2=502l^{2}=50), value adaptive to trajectory length, i.e. l=Lτl=L_{\tau} and l=1000/Lτl=1000/L_{\tau}, and the value output by Median Trick (MT) [18]. The training data is a series of quaternary [u,v,w,h][u,v,w,h], normalized to zero-mean before being fed into training. The results are shown in Table 9. Linear interpolation is simple but builds a strong baseline as it can stably improve the tracking performance concerning multiple metrics. Directly using GPR to interpolate the missing points hurts the performance and the results of GPR are not sensitive to the setting of ll.

There are two reasons preventing GPR from accurately interpolating missing segments. First, the trajectory is usually limited to at most hundreds of steps, providing very limited data points for GPR training to converge. On the other hand, the missing intermediate data points make the data series discontinuous, causing a huge challenge. We can fix the second issue by interpolating the trajectory with Linear Interpolation (LI) first and then smoothing the interpolated steps by GPR. This outperforms LI on DanceTrack but still regrades the performance by LI on MOT17. This is likely promoted by the non-linear motion on DanceTrack. By fixing the missing data issue of GPR, GPR can have a more accurate trajectory fitting over LI for the non-linear trajectory cases. But considering the outperforming from GPR is still minor compared with the Linear Interpolation-only version and GPR requires much heavier computation overhead, we do not recommend using such a practice in most multi-object tracking tasks. More careful and deeper study is still required on this problem.

| Tracker | HOTA↑\uparrow | MOTA↑\uparrow | IDF1↑\uparrow | FP(10410^{4})↓\downarrow | FN(10410^{4})↓\downarrow | IDs↓\downarrow | Frag↓\downarrow | AssA↑\uparrow | AssR↑\uparrow |
|---|---|---|---|---|---|---|---|---|---|
| CenterTrack [73] | - | 61.5 | 59.6 | 1.41 | 20.1 | 2,583 | - | - | - |
| QDTrack [42] | - | 64.6 | 65.1 | 1.41 | 18.3 | 2,652 | - | - | - |
| Lif_T [25] | 51.3 | 60.5 | 65.6 | 1.50 | 20.7 | 1,189 | 3,476 | 54.7 | 59.0 |
| TransCt [67] | 51.4 | 68.8 | 61.4 | 2.29 | 14.9 | 4,102 | 8,468 | 47.7 | 52.8 |
| TrackFormer [39] | - | 62.5 | 60.7 | 3.28 | 17.5 | 2,540 | - | - | - |
| OC-SORT | 52.4 | 58.2 | 65.1 | 0.44 | 23.0 | 784 | 2,006 | 57.6 | 63.5 |
| OC-SORT + LI | 52.9 | 59.4 | 65.7 | 0.66 | 22.2 | 801 | 1,030 | 57.5 | 63.9 |

| Tracker | HOTA↑\uparrow | MOTA↑\uparrow | IDF1↑\uparrow | FP(10410^{4})↓\downarrow | FN(10410^{4})↓\downarrow | IDs↓\downarrow | Frag↓\downarrow | AssA↑\uparrow | AssR↑\uparrow |
|---|---|---|---|---|---|---|---|---|---|
| MPNTrack [4] | 46.8 | 57.6 | 59.1 | 17.0 | 20.1 | 1,210 | 1,420 | 47.3 | 52.7 |
| TransCt [67] | 43.5 | 61.0 | 49.8 | 4.92 | 14.8 | 4,493 | 8,950 | 36.1 | 44.5 |
| ApLift [26] | 46.6 | 58.9 | 56.5 | 1.77 | 19.3 | 2,241 | 2,112 | 45.2 | 48.1 |
| TMOH [53] | 48.9 | 60.1 | 61.2 | 3.80 | 16.6 | 2,342 | 4,320 | 48.4 | 52.9 |
| LPC_MOT [13] | 49.0 | 56.3 | 62.5 | 1.17 | 21.3 | 1,562 | 1,865 | 52.4 | 54.7 |
| OC-SORT | 54.3 | 59.9 | 67.0 | 0.44 | 20.2 | 554 | 2,345 | 59.5 | 65.1 |
| OC-SORT + LI | 55.2 | 61.7 | 67.9 | 0.57 | 19.2 | 508 | 805 | 59.8 | 65.9 |

## Appendix C Results on More Benchmarks

When considering tracking in the crowd, focusing on only a part of the object can be beneficial [6] as it usually suffers less from occlusion than the full body. This line of study is conducted over hand tracking [40, 50], human pose [66] and head tracking [56, 2, 43] for a while. Moreover, with the knowledge of more fine-grained part trajectory, it can be useful in downstream tasks, such as action recognition [16, 17] and forecasting [31, 7, 33, 9]. As we are interested in the multi-object tracking in the crowd, we also evaluate the proposed OC-SORT on a recently proposed human head tracking dataset CroHD [56]. To make a fair comparison on only the association performance, we adopt OC-SORT by directing using the detections from existing tracking algorithms. The results are shown in Table 10. The detections of FairMOT [71] and HeadHunter [56] are extracted from their tracking results downloaded from the official leaderboard 22 2 https://motchallenge.net/results/Head_Tracking_21/. We use the same parameters for OC-SORT as on the other datasets. The results suggest a significant tracking performance improvement compared with the previous methods [56, 71] for human body part tracking. But the tracking performance is still relatively low (HOTA=∼\sim 40). It is highly related to the difficulty of having accurate detection of tiny objects. Some samples from the test set of HeadTrack are shown in the first two rows of Figure 6.

Although we use the same object detectors as some selected baselines, there is still variances in detections when compared with other methods. Therefore, we also report with the public detections on MOT17/MOT20 in Table 12 and Table 13. OC-SORT still outperforms the existing state-of-the-arts in the public tracking setting. And the outperforming of OC-SORT is more significant on MOT20 which has more severe occlusion scenes. Some samples from the test set of MOT20 are shown in the last row in Figure 6.

## Appendix D Pseudo-code of OC-SORT

See the pseudo-code of OC-SORT in Algorithm. 1.

## Appendix E More Results on DanceTrack

To gain more intuition about the improvement of OC-SORT over SORT, we provide more comparisons. In Figure 8, we show more samples where SORT suffers from ID switch or Fragmentation caused by non-linear motion or occlusion but OC-SORT survives. Furthermore, in Figure 9, we show more samples of trajectory visualizations from SORT and OC-SORT on DanceTrack-val set.

DanceTrack [54] is proposed to encourage better association algorithms instead of carefully tuning detectors. We train YOLOX [19] detector on MOT17 training set only to provide detections on DanceTrack. We find the tracking performance of OC-SORT is already higher than the baselines (Table 11). We believe the potential to improve multi-object tracking by better association strategy is still promising and DanceTrack is a good platform for the evaluation.

## Appendix F Integrate Appearance into OC-SORT

OC-SORT is pure motion-based but flexible to integrate with other association cues, such as object appearance. We make an attempt of adding appearance information into OC-SORT and achieve significant performance improvements, validated by experiments on MOT17, MOT20, and DanceTrack. Please refer to Deep OC-SORT [38] for details.

## Appendix G More Discussion of State Noise Sensitivity

In Section 3.2.1, we show that the noise of state estimate will be amplified to the noise of velocity estimate. This is because the velocity estimate is correlated to the state estimate. But the analysis is in a simplified model in which velocity itself does not gain noise from the transition directly and the noise of state estimate is i.i.d on different steps. However, in the general case, such a simplification does not hold. We now provide a more general analysis of the state noise sensitivity of SORT.

For the process in Eq 1, we follow the most commonly adapted implementation of Kalman filter 33 3 https://github.com/rlabbe/filterpy and SORT 44 4 https://github.com/abewley/sort for video multi-object tracking. Instead of writing the mean state estimate, we consider the noisy prediction of state estimate now, which is formulated as

|  | 𝐱t|t−1=𝐅t​𝐱t|t−1+𝐰t,{\displaystyle\mathbf{x}_{t|t-1}=\mathbf{F}_{t}\mathbf{x}_{t|t-1}+\mathbf{w}_{t}}, |  | (15) |
|---|---|---|---|

where 𝐰t\mathbf{w}_{t} is the process noise, drawn from a zero mean multivariate normal distribution, 𝒩{\mathcal{N}}, with covariance, 𝐰t∼𝒩⁡(0,𝐐t){\displaystyle\mathbf{w}_{t}\sim{\mathcal{N}}\left(0,\mathbf{Q}_{t}\right)}. As 𝐱t\mathbf{x}_{t} is a seven-tuple, i.e. 𝐱t=[u,v,s,r,u˙,v˙,s˙]⊤\mathbf{x}_{t}=[u,v,s,r,\dot{u},\dot{v},\dot{s}]^{\top}, the process noise applies to not just the state estimate but also the velocity estimates. Therefore, for a general form of analysis of temporal error magnification in Eq 5, we would get a different result because not just the position terms but also the velocity terms gain noise from the transition process. And the noise of velocity terms will amplify the noise of position estimate by the transition at the next step. We note the process noise as in practice:

|  | 𝐐t=[σu20000000σv20000000σs20000000σr20000000σu˙20000000σv˙20000000σs˙2],\mathbf{Q}_{t}=\begin{bmatrix}\sigma_{u}^{2}&0&0&0&0&0&0\\ 0&\sigma_{v}^{2}&0&0&0&0&0\\ 0&0&\sigma_{s}^{2}&0&0&0&0\\ 0&0&0&\sigma_{r}^{2}&0&0&0\\ 0&0&0&0&\sigma_{\dot{u}}^{2}&0&0\\ 0&0&0&0&0&\sigma_{\dot{v}}^{2}&0\\ 0&0&0&0&0&0&\sigma_{\dot{s}}^{2}\\ \end{bmatrix}, |  | (16) |
|---|---|---|---|

and the linear transition model as

|  | 𝐅t=[1000100010001000100010001000000010000000100000001].\mathbf{F}_{t}=\begin{bmatrix}1&0&0&0&1&0&0\\ 0&1&0&0&0&1&0\\ 0&0&1&0&0&0&1\\ 0&0&0&1&0&0&0\\ 0&0&0&0&1&0&0\\ 0&0&0&0&0&1&0\\ 0&0&0&0&0&0&1\\ \end{bmatrix}. |  | (17) |
|---|---|---|---|

We assume the time step when a track gets untracked is t1t_{1} and don’t consider the noise from previous steps. For simplicity, we assume the motion in the x-direction and y-direction do not correlate. We take the motion on the x-direction as an example without loss of generality:

|  | δut0∼𝒩⁡(0,σu2),δu˙t0∼𝒩⁡(0,σu˙2).\delta_{u_{t_{0}}}\sim\mathcal{N}(0,\sigma_{u}^{2}),\quad\delta_{\dot{u}_{t_{0}}}\sim\mathcal{N}(0,{\sigma_{\dot{u}}}^{2}). |  | (18) |
|---|---|---|---|

On the next step, with no correction from the observation, the error would be accumulated (Δ​t=1\Delta t=1),

|  | δut0+1∼𝒩⁡(0,2​σu2+σu˙2),δu˙t0+1∼𝒩⁡(0,2​σu˙2).\delta_{u_{t_{0}+1}}\sim\mathcal{N}(0,2\sigma_{u}^{2}+{\sigma_{\dot{u}}}^{2}),\quad\delta_{\dot{u}_{t_{0}+1}}\sim\mathcal{N}(0,{2\sigma_{\dot{u}}}^{2}). |  | (19) |
|---|---|---|---|

Therefore, the accumulation is even faster than we analyze in Section 3.2 as

|  | δut0+T∼𝒩⁡(0,(T+1)​σu2+12​T​(T+1)​σu˙2).\delta_{u_{t_{0}+T}}\sim\mathcal{N}(0,(T+1)\sigma_{u}^{2}+\frac{1}{2}T(T+1)\sigma_{\dot{u}}^{2}). |  | (20) |
|---|---|---|---|

In the practice of SORT, we have to suppress the noise from velocity terms because it is too sensitive. We achieve it by setting a proper value for the process noise 𝐐t\mathbf{Q}_{t}. For example, the most commonly adopted value 55 5 https://github.com/abewley/sort/blob/master/sort.py#L111 of 𝐐t\mathbf{Q}_{t} in SORT is

|  | 𝐐t=[100000001000000010000000100000000.0100000000.0100000000.0001].\mathbf{Q}_{t}=\begin{bmatrix}1&0&0&0&0&0&0\\ 0&1&0&0&0&0&0\\ 0&0&1&0&0&0&0\\ 0&0&0&1&0&0&0\\ 0&0&0&0&0.01&0&0\\ 0&0&0&0&0&0.01&0\\ 0&0&0&0&0&0&0.0001\\ \end{bmatrix}. |  | (21) |
|---|---|---|---|

In such a parameter setting, we have the ratio between the noise from position terms and velocity terms as

|  | β=(T+1)​σu20.5​T​(T+1)​σu˙2=200T.\beta=\frac{(T+1)\sigma_{u}^{2}}{0.5T(T+1)\sigma_{\dot{u}}^{2}}=\frac{200}{T}. |  | (22) |
|---|---|---|---|

In practice, a track is typically deleted if it keeps untracked for TdelT_{\text{del}} time steps. Usually we set Tdel<10T_{\text{del}}<10, so we have β>20\beta>20. Therefore, we usually consider the noise from velocity terms as secondary. Such a convention allows us to use the simplified model in Section 3.2.1 for noise analysis. But it also brings a side-effect that SORT can’t allow the velocity direction of a track to change quickly in a short time interval. We will see later (Section H) that it makes trouble to SORT when non-linear motion and occlusion come together and motivates the design of ORU in OC-SORT.

## Appendix H Intuition behind ORU

ORU is designed to fix the error accumulated during occlusion when an untracked track is re-associated with an observation. But in general, the bias in the state estimate 𝐱^\hat{\mathbf{x}} after being untracked for TT time steps can be fixed by the update stage once it gets re-associated with an observation. To be precise, the Optimal Kalman gain, i.e. 𝐊t\mathbf{K}_{t}, can use the re-associated observation to update the KF posteriori parameters. In general, such an expectation of KF’s behavior is reasonable. But because we usually set the corresponding covariance for velocity terms very small (Eq 21), it is difficult for SORT to steer to the correct velocity direction at the step of re-association.

Motivated by such observations, we design ORU. In the simplified model shown in Figure 7, the circle area with the shadow around each estimate footage is the eligible range to associate with observations inside. ORU is designed for the case that a track is re-associated after being untracked. Therefore, the typical situation is as shown in the figure that the true trajectory first goes away from the linear trajectory of KF estimates and then goes closer to it so that there can be a re-association. After the re-association, there would be a cross of the two trajectories.

In SORT, after re-associating with an observation, the direction of the velocity of the previously untracked track still has a significant difference from the true value. This is shown in Figure 7(b). This makes the estimate on the future steps lost again (the blue triangle). The reason is the convention of 𝐐\mathbf{Q} discussed in Appendix G. Therefore, even though the canonical KF can support fixing the accumulated error during being untracked theoretically, it is very rare in practice. In ORU, we follow the virtual trajectory where we have multiple virtual observations. In this way, even if the value of 𝐐[4:,4:]\mathbf{Q}[4:,4:] is small, we can still have a better-calibrated velocity direction after the time step t2t_{2}. We would like to note that the intuition behind ORU is from our observations in practice and based on the common convention of using Kalman filter for multi-object tracking. It does not make fundamental changes to upgrade the power of the canonical Kalman filter.

Here we provide a more formal mathematical expression to compare the behaviors of SORT and OC-SORT. Assume that the track was lost at the time step t1t_{1} and re-associated at t2t_{2}. We assume the mean state estimate is

|  | 𝐱^t1|t1=[u1,v1,s1,r1,u˙1,v˙1,s˙1]⊤,\hat{\mathbf{x}}_{t_{1}|t_{1}}=[u_{1},v_{1},s_{1},r_{1},\dot{u}_{1},\dot{v}_{1},\dot{s}_{1}]^{\top}, |  | (23) |
|---|---|---|---|

and the covariance at t1t_{1} is

|  | 𝐏t1|t1=[σu120000000σv120000000σs120000000σr120000000σu˙120000000σv˙120000000σs˙12].\mathbf{P}_{t_{1}|t_{1}}=\begin{bmatrix}\sigma_{u_{1}}^{2}&0&0&0&0&0&0\\ 0&\sigma_{v_{1}}^{2}&0&0&0&0&0\\ 0&0&\sigma_{s_{1}}^{2}&0&0&0&0\\ 0&0&0&\sigma_{r_{1}}^{2}&0&0&0\\ 0&0&0&0&\sigma_{\dot{u}_{1}}^{2}&0&0\\ 0&0&0&0&0&\sigma_{\dot{v}_{1}}^{2}&0\\ 0&0&0&0&0&0&\sigma_{\dot{s}_{1}}^{2}\\ \end{bmatrix}. |  | (24) |
|---|---|---|---|

Then, because the covariance expands from the input of process noise at each step of predict, at t2t_{2}, we have the priori estimates (tΔ=t2−t1t_{\Delta}=t_{2}-t_{1}) of state

|  | 𝐱^t2|t2−1=[u2,v2,s2,r2,u˙2,v˙2,s˙2]⊤,\hat{\mathbf{x}}_{t_{2}|t_{2}-1}=[u_{2},v_{2},s_{2},r_{2},\dot{u}_{2},\dot{v}_{2},\dot{s}_{2}]^{\top}, |  | (25) |
|---|---|---|---|

with

|  | u2\displaystyle u_{2} | =u1+u˙1​tΔ,\displaystyle=u_{1}+\dot{u}_{1}t_{\Delta}, |  | (26) |
|---|---|---|---|---|
|  | v2\displaystyle v_{2} | =v1+v˙1​tΔ,\displaystyle=v_{1}+\dot{v}_{1}t_{\Delta}, |  |  |
|  | s2\displaystyle s_{2} | =s1+s˙1​tΔ,\displaystyle=s_{1}+\dot{s}_{1}t_{\Delta}, |  |  |
|  | r2\displaystyle r_{2} | =r1,\displaystyle=r_{1}, |  |  |
|  | u˙2\displaystyle\dot{u}_{2} | =u˙1,\displaystyle=\dot{u}_{1}, |  |  |
|  | v˙2\displaystyle\dot{v}_{2} | =v˙1,\displaystyle=\dot{v}_{1}, |  |  |
|  | s˙2\displaystyle\dot{s}_{2} | =s˙1.\displaystyle=\dot{s}_{1}. |  |  |

And the priori covariance

|  | 𝐏t2|t2−1=[σu220000000σv220000000σs220000000σr220000000σu˙220000000σv˙220000000σs˙22],\mathbf{P}_{t_{2}|t_{2}-1}=\begin{bmatrix}\sigma_{u_{2}}^{2}&0&0&0&0&0&0\\ 0&\sigma_{v_{2}}^{2}&0&0&0&0&0\\ 0&0&\sigma_{s_{2}}^{2}&0&0&0&0\\ 0&0&0&\sigma_{r_{2}}^{2}&0&0&0\\ 0&0&0&0&\sigma_{\dot{u}_{2}}^{2}&0&0\\ 0&0&0&0&0&\sigma_{\dot{v}_{2}}^{2}&0\\ 0&0&0&0&0&0&\sigma_{\dot{s}_{2}}^{2}\\ \end{bmatrix}, |  | (27) |
|---|---|---|---|

with

|  | σu22\displaystyle\sigma_{u_{2}}^{2} | =σu12+tΔ​(σu2+σu˙12)+tΔ​(tΔ−1)2​σu˙2,\displaystyle=\sigma_{u_{1}}^{2}+t_{\Delta}(\sigma_{u}^{2}+\sigma_{\dot{u}_{1}}^{2})+\frac{t_{\Delta}(t_{\Delta}-1)}{2}\sigma_{\dot{u}}^{2}, |  | (28) |
|---|---|---|---|---|
|  | σv22\displaystyle\sigma_{v_{2}}^{2} | =σv12+tΔ​(σv2+σv˙12)+tΔ​(tΔ−1)2​σv˙2,\displaystyle=\sigma_{v_{1}}^{2}+t_{\Delta}(\sigma_{v}^{2}+\sigma_{\dot{v}_{1}}^{2})+\frac{t_{\Delta}(t_{\Delta}-1)}{2}\sigma_{\dot{v}}^{2}, |  |  |
|  | σs22\displaystyle\sigma_{s_{2}}^{2} | =σs12+tΔ​(σs2+σs˙12)+tΔ​(tΔ−1)2​σs˙2,\displaystyle=\sigma_{s_{1}}^{2}+t_{\Delta}(\sigma_{s}^{2}+\sigma_{\dot{s}_{1}}^{2})+\frac{t_{\Delta}(t_{\Delta}-1)}{2}\sigma_{\dot{s}}^{2}, |  |  |
|  | σr22\displaystyle\sigma_{r_{2}}^{2} | =σr12+tΔ​σr2,\displaystyle=\sigma_{r_{1}}^{2}+t_{\Delta}\sigma_{r}^{2}, |  |  |
|  | σu˙22\displaystyle\sigma_{\dot{u}_{2}}^{2} | =σu˙12+tΔ​σu˙2,\displaystyle=\sigma_{\dot{u}_{1}}^{2}+t_{\Delta}\sigma^{2}_{\dot{u}}, |  |  |
|  | σv˙22\displaystyle\sigma_{\dot{v}_{2}}^{2} | =σv˙12+tΔ​σv˙2,\displaystyle=\sigma_{\dot{v}_{1}}^{2}+t_{\Delta}\sigma^{2}_{\dot{v}}, |  |  |
|  | σs˙22\displaystyle\sigma_{\dot{s}_{2}}^{2} | =σs˙12+tΔ​σs˙2.\displaystyle=\sigma_{\dot{s}_{1}}^{2}+t_{\Delta}\sigma^{2}_{\dot{s}}. |  |  |

Now, SORT will keep going forward as normal. Therefore, with the re-associated observation 𝐳t2\mathbf{z}_{t_{2}}, we have

|  | SORT{𝐱^t2|t2=𝐱^t2|t2−1+𝐊t2​(𝐳t2−𝐇​𝐱^t2|t2−1),𝐏t2|t2=(𝐈−𝐊t2𝐇)𝐏t2|t2−1\text{{SORT}}\left\{\begin{aligned} \hat{\mathbf{x}}_{t_{2}|t_{2}}&=\hat{\mathbf{x}}_{t_{2}|t_{2}-1}+\mathbf{K}_{t_{2}}(\mathbf{z}_{t_{2}}-\mathbf{H}\hat{\mathbf{x}}_{t_{2}|t_{2}-1}),\\ \mathbf{P}_{t_{2}|t_{2}}&=(\mathbf{I}-\mathbf{K}_{t_{2}}\mathbf{H}_{)}\mathbf{P}_{t_{2}|t_{2}-1}\end{aligned}\right. |  | (29) |
|---|---|---|---|

where the observation model is

|  | 𝐇=[1000000010000000100000001000],\mathbf{H}=\begin{bmatrix}1&0&0&0&0&0&0\\ 0&1&0&0&0&0&0\\ 0&0&1&0&0&0&0\\ 0&0&0&1&0&0&0\\ \end{bmatrix}, |  | (30) |
|---|---|---|---|

and the Kalman gain is

|  | 𝐊t2=𝐏t2|t2−1​𝐇⊤​(𝐇𝐏t2|t2−1​𝐇⊤+𝐑t2)−1.\mathbf{K}_{t_{2}}=\mathbf{P}_{t_{2}|t_{2}-1}\mathbf{H}^{\top}(\mathbf{H}\mathbf{P}_{t_{2}|t_{2}-1}\mathbf{H}^{\top}+\mathbf{R}_{t_{2}})^{-1}. |  | (31) |
|---|---|---|---|

On the other hand, OC-SORT will replay Kalman filter predict on a generated virtual trajectory to gain the posteriori estimates on t2t_{2} (ORU). With the default linear motion analysis, we have the virtual trajectory as

|  | 𝐳~t=𝐳t1+t−t1t2−t1​(𝐳t2−𝐳t1),t1<t<t2.\tilde{\mathbf{z}}_{t}=\mathbf{z}_{t_{1}}+\frac{t-t_{1}}{t_{2}-t_{1}}(\mathbf{z}_{t_{2}}-\mathbf{z}_{t_{1}}),t_{1}<t<t_{2}. |  | (32) |
|---|---|---|---|

Now, to derive the posteriori estimate, we will run the loop between predict and re-update from t1t_{1} to t2t_{2}.

|  | OC-SORT{𝐱^t|t=𝐅​𝐱^t−1|t−1+𝐊t​(𝐳~t−𝐇𝐅​𝐱^t−1|t−1)𝐏t|t=(𝐈−𝐊t​𝐇)​(𝐅𝐏t−1|t−1​𝐅⊤+𝐐t)\text{{OC-SORT}}\left\{\begin{aligned} \hat{\mathbf{x}}_{t|t}&=\mathbf{F}\hat{\mathbf{x}}_{t-1|t-1}+\mathbf{K}_{t}(\tilde{\mathbf{z}}_{t}-\mathbf{H}\mathbf{F}\hat{\mathbf{x}}_{t-1|t-1})\\ \mathbf{P}_{t|t}&=(\mathbf{I}-\mathbf{K}_{t}\mathbf{H})(\mathbf{F}\mathbf{P}_{t-1|t-1}\mathbf{F}^{\top}+\mathbf{Q}_{t})\end{aligned}\right. |  | (33) |
|---|---|---|---|

where the Kalman gain is

|  | 𝐊t=𝐏t|t−1​𝐇t⊤​(𝐇𝐏t|t−1​𝐇⊤+𝐑t)−1,\mathbf{K}_{t}=\mathbf{P}_{t|t-1}\mathbf{H}_{t}^{\top}(\mathbf{H}\mathbf{P}_{t|t-1}\mathbf{H}^{\top}+\mathbf{R}_{t})^{-1}, |  | (34) |
|---|---|---|---|

and we can always rewrite it with

|  | 𝐏t|t−1=𝐅𝐏t−1|t−1​𝐅⊤+𝐐t.\mathbf{P}_{t|t-1}=\mathbf{F}\mathbf{P}_{t-1|t-1}\mathbf{F}^{\top}+\mathbf{Q}_{t}. |  | (35) |
|---|---|---|---|

In the common practice of Kalman filter, we assume a constant set of Gaussian noise for the process noise 𝐐t\mathbf{Q}_{t}. This assumption typically can’t hold in practice. This makes the conflict that when there are consistent observations over time, we require a small process noise for multi-object tracking in high-frame-rate videos. However, when there is a period of observation missing, the direction difference between the true direction and the direction maintained by the linear motion assumption grows. This causes the failure of SORT to consistently track previously lost targets even after re-association.

We show the different outcomes of SORT and OC-SORT upon re-associating lost targets in Eq 29 and Eq 33. Analyzing their difference more deeply will require more assumptions of the underlying true object trajectory and the observations. Therefore, instead of theoretical proof, we demonstrate the gain of performance from OC-SORT over SORT empirically as shown in the experiments.
