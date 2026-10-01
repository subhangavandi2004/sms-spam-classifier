"""SMS spam classification: TF-IDF with Naive Bayes, Logistic Regression and Linear SVM."""
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

SEED = 42


def make(clf):
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                               lowercase=True)), ("clf", clf)])


def main():
    df = pd.read_csv("data/sms.tsv", sep="\t", header=None, names=["label", "text"])
    y = (df["label"] == "spam").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(df["text"], y, test_size=0.2, stratify=y,
                                              random_state=SEED)
    models = {
        "naive_bayes": make(MultinomialNB(alpha=0.1)),
        "logistic_regression": make(LogisticRegression(max_iter=2000, C=10, class_weight="balanced")),
        "linear_svm": make(LinearSVC(C=1.0, class_weight="balanced")),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    results = {"n_messages": len(df), "spam_share_pct": round(float(y.mean() * 100), 1),
               "always_predict_ham_accuracy_pct": round(float((1 - y.mean()) * 100), 1)}
    for name, m in models.items():
        cv_f1 = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="f1").mean()
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        results[name] = {"cv_f1_spam": round(float(cv_f1), 4),
                         "test_f1_spam": round(float(f1_score(y_te, pred)), 4),
                         "test_precision": round(float(precision_score(y_te, pred)), 4),
                         "test_recall": round(float(recall_score(y_te, pred)), 4)}
    best = max(models, key=lambda k: results[k]["cv_f1_spam"])  # chosen on training CV only
    results["best_model_by_cv_f1"] = best
    pred = models[best].predict(X_te)

    errors = X_te.to_frame("text").assign(actual=y_te.values, predicted=pred)
    errors = errors[errors["actual"] != errors["predicted"]]
    errors["error_type"] = errors.apply(
        lambda r: "missed spam" if r["actual"] == 1 else "false alarm", axis=1)
    errors.to_csv("misclassified_test_messages.csv", index=False)
    results["errors"] = errors["error_type"].value_counts().to_dict()

    joblib.dump(models[best], "model.joblib")
    json.dump(results, open("results.json", "w"), indent=2)
    print(json.dumps(results, indent=2))
    print(classification_report(y_te, pred, target_names=["ham", "spam"]))


if __name__ == "__main__":
    main()
