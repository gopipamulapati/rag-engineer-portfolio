from app.retrieval import content_words, stem


def test_stemming_matches_word_forms():
    assert stem("backs") == stem("backed") == "back"
    assert stem("acknowledge") == stem("acknowledged")
    assert stem("queries") == stem("query")
    assert stem("access") == "access"  # double-s words are left alone


def test_content_words_drop_stopwords_and_split_hyphens():
    assert content_words("What is the on-call rotation?", stemmed=False) == ["call", "rotation"]


def test_search_finds_the_right_document(retriever):
    hits = retriever.search("How long are ClickHouse backups retained?")
    assert hits[0].chunk.source == "data_retention_policy.txt"
    assert "14 days" in hits[0].chunk.text


def test_search_with_only_stopwords_returns_nothing(retriever):
    assert retriever.search("what is it?") == []
