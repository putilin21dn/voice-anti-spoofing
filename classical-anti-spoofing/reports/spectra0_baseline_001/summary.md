# Spectra-0 baseline

Порог 1.02 не подбирался заново. Score — средний зазор logit, не вероятность.

## n100

На этом наборе выбирался порог Spectra-0. Метрики здесь не являются независимым тестом.

- calls: 190
- human: 91
- machine: 99
- recall_synthetic: 0.7070707070707071
- specificity_human: 0.8681318681318682
- precision_synthetic: 0.8536585365853658
- f1_synthetic: 0.7734806629834254
- balanced_accuracy: 0.7876012876012877
- fpr: 0.13186813186813187
- fnr: 0.29292929292929293
- roc_auc: 0.8263958263958264
- pr_auc: 0.8683374859260062

## test20

Независимый test20. Порог на нём не подбирался.

- calls: 43
- human: 18
- machine: 25
- recall_synthetic: 0.68
- specificity_human: 0.9444444444444444
- precision_synthetic: 0.9444444444444444
- f1_synthetic: 0.7906976744186046
- balanced_accuracy: 0.8122222222222222
- fpr: 0.05555555555555555
- fnr: 0.32
- roc_auc: 0.8466666666666667
- pr_auc: 0.9150192173597748
