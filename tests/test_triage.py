from triage.classifier import FRAUD_PATTERNS, load_rows, template_key


def _matches_fraud_patterns(text: str) -> bool:
    return any(pattern.search(text) for pattern in FRAUD_PATTERNS)


def test_unrecognized_transfer_phrasing_is_caught():
    # The gap found by inspecting real false negatives under the full deployed policy:
    # no existing keyword matched "a transfer/transaction I did not make", and the
    # model's probability alone wasn't reliably crossing the threshold for it either.
    assert _matches_fraud_patterns("There's a suspicious ETH transfer to an unknown wallet that I did not make.")
    assert _matches_fraud_patterns("I see a trade on my account that I never made.")
    assert _matches_fraud_patterns("This withdrawal was never authorized by me.")


def test_similar_non_fraud_phrasing_does_not_collide():
    # Near-miss phrasings from the real training data that must NOT trigger the rule,
    # since they use different verbs ("received"/"sends"/"arrives") for an unrelated
    # account-access/transaction-dispute problem, not an unrecognized transaction.
    assert not _matches_fraud_patterns("My withdrawal of 2 ETH shows completed but I never received it, please reverse it.")
    assert not _matches_fraud_patterns("Verification keeps failing during login, it never sends me the code.")
    assert not _matches_fraud_patterns("The password reset email never arrives, I've checked spam.")


def test_load_rows_matches_known_corpus_shape():
    texts, labels = load_rows()
    assert len(texts) == 400
    assert len(texts) == len(labels)
    assert set(labels) == {"account-access", "transaction-dispute", "fraud-report", "general"}


def test_template_key_collapses_near_duplicates():
    a = template_key("Hi, What's the minimum amount I can buy of Bitcoin?")
    b = template_key("Hello team, What's the minimum amount I can buy of Ethereum? Let me know.")
    assert a == b
