# Result of trainings

## Dataset 1

Command: `make train-dataset1`
Dataset: dataset/sqli_dataset1.csv
Model: sqli_model/1

```bash
> python train.py
773/773 [==============================] - 364s 467ms/step - loss: 0.0702 - accuracy: 0.9749
Epoch 2/10
773/773 [==============================] - 365s 472ms/step - loss: 0.0183 - accuracy: 0.9957
Epoch 3/10
773/773 [==============================] - 400s 518ms/step - loss: 0.0135 - accuracy: 0.9971
Epoch 4/10
773/773 [==============================] - 423s 547ms/step - loss: 0.0123 - accuracy: 0.9972
Epoch 5/10
773/773 [==============================] - 436s 564ms/step - loss: 0.0106 - accuracy: 0.9977
Epoch 6/10
773/773 [==============================] - 337s 436ms/step - loss: 0.0103 - accuracy: 0.9978
Epoch 7/10
773/773 [==============================] - 335s 433ms/step - loss: 0.0096 - accuracy: 0.9981
Epoch 8/10
773/773 [==============================] - 332s 430ms/step - loss: 0.0094 - accuracy: 0.9980
Epoch 9/10
773/773 [==============================] - 385s 499ms/step - loss: 0.0096 - accuracy: 0.9981
Epoch 10/10
773/773 [==============================] - 426s 551ms/step - loss: 0.0093 - accuracy: 0.9981
194/194 [==============================] - 8s 42ms/step
Accuracy: 62.95%
Recall: 0.00%
Precision: 100.00%
F1-score: 0.00%
Specificity: 100.00%
ROC: 0.00%
```

## Dataset 2

Command: `make train-dataset2`
Dataset: dataset/sqli_dataset2.csv
Model: sqli_model/2

```bash
789/789 [==============================] - 377s 474ms/step - loss: 0.0603 - accuracy: 0.9801
Epoch 2/11
789/789 [==============================] - 367s 465ms/step - loss: 0.0139 - accuracy: 0.9969
Epoch 3/11
789/789 [==============================] - 383s 485ms/step - loss: 0.0104 - accuracy: 0.9983
Epoch 4/11
789/789 [==============================] - 360s 456ms/step - loss: 0.0083 - accuracy: 0.9986
Epoch 5/11
789/789 [==============================] - 347s 440ms/step - loss: 0.0075 - accuracy: 0.9988
Epoch 6/11
789/789 [==============================] - 341s 432ms/step - loss: 0.0077 - accuracy: 0.9988
Epoch 7/11
789/789 [==============================] - 356s 451ms/step - loss: 0.0071 - accuracy: 0.9988
Epoch 8/11
789/789 [==============================] - 344s 437ms/step - loss: 0.0072 - accuracy: 0.9989
Epoch 9/11
789/789 [==============================] - 346s 438ms/step - loss: 0.0070 - accuracy: 0.9989
Epoch 10/11
789/789 [==============================] - 355s 450ms/step - loss: 0.0072 - accuracy: 0.9988
Epoch 11/11
789/789 [==============================] - 358s 454ms/step - loss: 0.0071 - accuracy: 0.9989
198/198 [==============================] - 7s 34ms/step
Accuracy: 63.96%
Recall: 0.00%
Precision: 100.00%
F1-score: 0.00%
Specificity: 100.00%
ROC: 0.00%
```
