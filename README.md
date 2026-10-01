# SMS Spam Classifier

TF-IDF (word 1-2 grams) with three classic models, plus a FastAPI endpoint and a manual look at the errors.

**Data:** 5,572 SMS messages, 13.4% spam (https://github.com/justmarkham/DAT7, `sms.tsv`).

**Run:** `pip install -r requirements.txt`, `python train.py`, `pytest test_api.py`, `uvicorn app:app`.

## Results (spam class, 20% held-out test set of 1,115 messages)

A model that always says "not spam" is already 86.6% accurate, so I report F1 on the spam class.

| Model | CV F1 | Test F1 | Precision | Recall |
|---|---|---|---|---|
| Naive Bayes | 0.950 | 0.943 | 1.000 | 0.893 |
| **Logistic Regression** (chosen by CV) | **0.951** | **0.952** | 0.972 | 0.933 |
| Linear SVM | 0.951 | 0.959 | 0.979 | 0.940 |

The chosen model made 14 mistakes on 1,115 messages: 10 missed spam and 4 false alarms
(see `misclassified_test_messages.csv`). Most missed spam are very short or look like normal chat
(for example a bare "ringtoneking 84484"), which word counts alone cannot catch.

The three models are within noise of each other on this test set.
