# Dataset: KDD Cup 1999

## About

The **KDD Cup 1999 dataset** is the most widely used benchmark for network intrusion
detection systems. It was created for the Third International Knowledge Discovery and
Data Mining Tools Competition held in conjunction with KDD-1999.

Each record represents a network connection and is labeled as either **normal** traffic
or one of **22 attack types** grouped into four categories:

| Category | Description                     | Example Attacks              |
|----------|---------------------------------|------------------------------|
| DoS      | Denial of Service               | smurf, neptune, back, land   |
| Probe    | Surveillance / port scanning    | portsweep, ipsweep, nmap     |
| R2L      | Remote to Local (unauthorized)  | warezclient, guess_passwd    |
| U2R      | User to Root (privilege escalation) | buffer_overflow, rootkit |

## Dataset Stats (10% subset used in this lab)

- **Total samples:** ~494,021
- **Features:** 41 (38 numeric + 3 categorical)
- **Categorical features:** `protocol_type`, `service`, `flag`
- **Task in this lab:** Binary classification (Normal vs Attack)

## How We Use It

Our `train_model.py` script uses `sklearn.datasets.fetch_kddcup99` which **automatically
downloads and caches** the dataset on first run. No manual download is needed.

The cached data is stored at: `~/scikit_learn_data/kddcup99-py3/`

## References

- [KDD Cup 1999 - UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/kdd+cup+1999+data)
- [sklearn.datasets.fetch_kddcup99 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_kddcup99.html)
- Tavallaee, M., et al. "A detailed analysis of the KDD CUP 99 data set." (2009)
