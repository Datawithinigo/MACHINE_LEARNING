# Intelligible Models for HealthCare: Predicting Pneumonia Risk 

## Methodological revision 

**Dataset and problem** 

The paper addresses two clinical prediction tasks: 

- The first is pneumonia mortality risk: given a patient admitted with pneumonia, predict their probability of death (POD) so clinicians can decide between hospitalization and outpatient treatment.   
- The second is 30-day hospital readmission: predict whether a recently discharged patient will need to return within 30 days, a metric tied to hospital quality and financial penalties.

*Pneumonia dataset*. Drawn from a large multi-institutional 1990s study (Cooper et al., 1997), it contains 14,199 adult patients (9,847 train / 4,352 test, a 70:30 split). There are 46 features spanning four categories: patient history (age, gender, comorbidities such as asthma, diabetes, cancer), physical examination findings (heart rate, respiration rate, temperature, blood pressure), laboratory results (BUN, WBC, albumin, creatinine, pH, pO₂), and chest X-ray findings (pleural effusion, lung collapse). Two notable challenges are present. First, class imbalance: only 10.86% of patients died, creating a heavily skewed target. Second, missing data encoded as zero: for many continuous variables (heart rate, BUN, glucose), a value of 0 does not mean zero but rather that the measurement was not taken or was assumed normal — a common but analytically problematic convention in clinical datasets. The authors observe this explicitly for heart rate, noting that 91% of patients have a recorded value of 0\.

A critical data bias challenge also affects this dataset: patients with asthma who presented with pneumonia were routinely sent directly to the ICU, received aggressive care, and consequently had lower observed mortality. Any model trained on this data therefore learns the spurious rule "asthma → lower risk," which is clinically dangerous if deployed.

*30-day readmission dataset*. Far larger and more modern, it comes from a collaboration with a major hospital and covers 195,901 training patients (2011–2012) and 100,823 test patients (2013), with 3,956 features per patient including lab results, summaries of physician notes, and full hospitalization history. The class imbalance is similar: 8.91% of patients are readmitted within 30 days. The sheer feature dimensionality (nearly 4,000 variables) makes full model inspection impractical, and the authors shift from examining the entire model to examining per-patient explanations for representative cases.

**Some changes that could be nice to implement or consider (version1: es un poco mas resumida y general, de la que hice antes de contrastar realmente con el contenido del curso)**

Implement a formal fairness audit across demographic subgroups, combined with disaggregated performance evaluation: 

The paper evaluates model performance using a single aggregate AUC figure — 0.857 for GA²M on pneumonia and 0.783 on readmission. While these numbers are competitive with Random Forests and LogitBoost (Table 2), they say nothing about *who* the model fails on. This is a critical gap in a healthcare setting.

Aggregate AUC can mask substantial performance disparities across subgroups defined by age, sex, or socioeconomic proxies. A model that achieves AUC \= 0.86 overall might perform at 0.91 for younger patients and 0.74 for elderly patients — a difference that has direct clinical consequences given that treatment decisions are made at the individual level. This problem is well-documented under the framework of algorithmic fairness, specifically the notion of *equalized odds* (Hardt et al., 2016), which requires that a classifier's true positive rate and false positive rate be equal across protected groups. Violations of equalized odds in clinical risk models have been empirically demonstrated: Obermeyer et al. (2019) showed that a widely used hospital readmission algorithm systematically underestimated the severity of illness in Black patients relative to white patients, because it used healthcare cost as a proxy for health need — and Black patients historically incurred lower costs for the same level of illness due to reduced access to care.

This connects directly to the paper's own motivation. The authors are already sensitive to data bias: the asthma finding is precisely a case where a model learned a pattern that reflects differential treatment rather than differential underlying risk. However, they treat this as a one-off, manually detectable problem rather than as an instance of a broader systematic issue. A formal subgroup analysis would operationalize this concern: for each demographic stratum available in the data (age bands, sex, comorbidity burden), one would compute calibration curves, AUC, and importantly the false negative rate (patients predicted low-risk but who died or were readmitted), since false negatives in triage are the most dangerous errors.

The GA²M architecture is particularly well-suited for this extension. Because the model is additive and modular, one could inspect whether specific shape functions — for instance, the age term or the asthma term — behave differently when the model is retrained on demographic subsets, or whether interaction terms implicitly encode demographic proxies. This is harder to do with a black-box ensemble. The intelligibility that the authors rightly celebrate as GA²M's core advantage becomes even more valuable when it is used not just to validate the overall model but to audit it for equity.

In terms of evaluation methodology, It would be a good idea also complement AUC with calibration metrics such as the Brier score and reliability diagrams. AUC measures discriminative ability (ranking), but clinical decision-making depends on well-calibrated probabilities — a prediction of p \= 0.8 should mean roughly 80% of such patients actually experience the outcome. This is especially important for the readmission task, where downstream interventions (follow-up calls, extended stays) are allocated based on predicted probability thresholds.

**References**

Cooper, G. F., Aliferis, C. F., Ambrosino, R., Aronis, J., Buchanan, B. G., Caruana, R., Fine, M. J., Glymour, C., Gordon, G., Hanusa, B. H., Janosky, J. E., Meek, C., Mitchell, T., Richardson, T., & Spirtes, P. (1997). An evaluation of machine-learning methods for predicting pneumonia mortality. *Artificial intelligence in medicine*, *9*(2), 107–138. https://doi.org/10.1016/s0933-3657(96)00367-3   
Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. *Advances in Neural Information Processing Systems (NeurIPS), 29*.

| [https://doi.org/10.48550/arXiv.1610.02413](https://doi.org/10.48550/arXiv.1610.02413) |
| :---- |

Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science, 366*(6464), 447–453. [https://doi.org/10.1126/science.aax2342](https://doi.org/10.1126/science.aax2342)

Hastie, T., & Tibshirani, R. (1990). *Generalized Additive Models*. Chapman & Hall/CRC.  
Lou, Y., Caruana, R., & Gehrke, J. (2012). Intelligible models for classification and regression. *KDD 2012*.  
[https://doi.org/10.1201/9780203753781](https://doi.org/10.1201/9780203753781) 

Lou, Y., Caruana, R., Gehrke, J., & Hooker, G. (2013). Accurate intelligible models with pairwise interactions. *KDD 2013*.  
DOI:[10.1145/2487575.2487579](https://doi.org/10.1145/2487575.2487579) 

**Some changes that could be nice to implement or consider (version 2\)**   
Lo dividiria en 3 steps que se podrian implementar con tal de hacer una revisión de la metodologia del artículo **(si os parece bien lo redacto, o lo resumo, o si lo queréis mirar vosotros y añadir algo más, es una seccion que se puede añadir varias cosas creo, o podemos escoger las más importantes y el resto emncionarlas solo, cómo lo veis? )**.:  

1) ***Clustering by GMM and Hierarchical \-\> Systematic Subpopulation Discovery via Unsupervised Clustering.*** 

The paper’s most celebrated finding \- the asthma paradox \- was discovered “by accident” through rule-based inspection. The authors acknowledge that other hidden biases (chronic lung disease, history of chest pain) may exist but offer no systematic way to find them. This is the core weakness of their validation approach.   
We would consider to introduce a pre-modeling clustering phase using Gaussian Mixture Models (GMM) on the feature space before training any GA2M. As covered in the course slides (Clustering: Hierarchical and GMM), GMMs are particularly suited to healthcare data because, unlike K-means, they allow overlapping soft-cluster membership \- a patient can belong partially to a “severe but well-treated” cluster and partially to a “moderate illness” cluster, which is clinically realistic.   
In fact, the specific procedure would be: 1\) Fit a GMM on patient features using BIC to select the number of components; 2\) For each discovered cluster, compute the mean risk score and mean treatment intensity (ICU admission rate, number of medications administered); 3\) Flag any cluster where treatment instensity is high but observed mortality is low \- precisely the signature of the asthma paradox.   
This transforms a one-off expert observation into a **repeatable audit**. Du et al. (2021) demonstrate that unsupervised subgroup discovery before supervised training prevents the systematic "averaging out" of risks in heterogeneous clinical populations, where a single aggregate risk function conceals fundamentally different underlying processes. In the pneumonia context, this would have surfaced not only asthmatic patients but potentially pregnant women and patients with chronic lung disease — groups the authors themselves speculate about but cannot confirm.   
REF para aqui: Du, M., Yang, F., Zou, N., & Hu, X. (2021). Fairness in deep learning: A computational perspective. *IEEE Intelligent Systems, 36*(4), 25–34. *(Stage 1 — subgroup discovery)DOI:* [10.1109/MIS.2020.3000681](https://doi.org/10.1109/MIS.2020.3000681)

2) ***Feature Selection and Regularization: Principled Feature Selection in High-Dimensional Scenarios*** 

The 30-day readmission dataset contains 3,956 features. The authors rely on GA²M's internal cross-validation to select *k* pairwise interactions, but perform **no explicit pre-screening and this exposes the model to two well-documented risks: redundancy (multiple visit-count windows capturing the same underlying construct) and spurious correlation driven by the curse of dimensionality.**   
So that, It would be nice to implement an **Elastic Net pre-filte** before passing features to GA²M. The Elastic Net is preferable to pure LASSO here because it handles correlated feature groups more gracefully — retaining one representative from a correlated cluster rather than arbitrarily selecting among them. 

3) **Supervised LEarning Evaluation: Moving from AUC Ranking to Clinical Calibration**

The paper evaluates every model — logistic regression, Random Forests, LogitBoost, GAM, GA²M — using a single AUC figure (Table 2). AUC measures a model's ability to *rank* patients by risk, but clinical decisions are threshold-based: a predicted probability above some value triggers ICU admission or an extended stay.   
AUC and calibration are orthogonal properties — a model can achieve high AUC while being systematically miscalibrated.   
Therefore it would be nice to complement the only-AUC evaluation with other metrics framework, for instance; 1\) Reliability diagrams (calibration curves), that is plot mean predicted probability against observed outcome frequency in decile bins. A well-calibrated model lies on the diagonal; 2\) Brier Score: the mean squared error between predicted probability and binary outcome and 3\) Expected Calibration Error (ECE)that is a scalar summary of the reliability diagram, useful for comparing models.

This matters most for the pneumonia task, where the decision is binary and high-stakes: admit to hospital or treat as outpatient. 

A model predicting p \= 0.85 for a patient who is then sent home because the true risk was 0.3 is a direct patient safety failure that AUC alone would not detect. 

Moreover, calibration analysis stratified by the GMM clusters identified in Stage 1 would reveal whether miscalibration is concentrated in specific subpopulations — precisely the kind of structured failure mode the paper found manually with the asthma group.

4) **Time Series Analysis: Temporal Robustness via Rolling-Window Validation** 

The readmission model is trained on 2011–2012 data and tested on 2013 data — a clean temporal split. However, the authors treat this as a standard train/test evaluation and do not examine whether the model's shape functions are stable over time. 

We know that healthcare data is inherently subject to concept drift: clinical protocols, antibiotic guidelines, discharge criteria, and coding practices change from year to year, meaning a feature's relationship to the outcome can shift even if the feature itself does not. 

**(Esto tambien se podria explicar más, pero de momento lo dejo así)** 

